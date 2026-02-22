from __future__ import annotations

import types

import pytest
from bs4 import BeautifulSoup
from app import routes
from app.cache_manager import CacheManager
from app.fetch_utils import FetchUtils
from app.index.builder import DummyEmbeddingBackend
from app.index.models import Chunk
from app.index.vector_store import LocalVectorStore
from app.language_utils import LanguageContext, NoOpTranslator
from app.match_engine import StaffProfile
from app.rag.retriever import EmbeddingRetriever, RetrievalResult


@pytest.mark.asyncio
async def test_queue_head_returns_204(client) -> None:
    response = await client.head("/queue")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_analyze_topic_returns_themes(client) -> None:
    response = await client.post(
        "/analyze-topic",
        json={"text": "Psykologi og helse ved Oslo Nye Høyskole."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["themes"]
    assert "normalizedPreview" in data


@pytest.mark.asyncio
async def test_analyze_topic_rejects_empty_text(client) -> None:
    response = await client.post("/analyze-topic", json={"text": "   "})
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Mangler tematisk innhold. Skriv tekst for tema."


@pytest.mark.asyncio
async def test_match_endpoint_returns_ranked_result(client) -> None:
    response = await client.post(
        "/match",
        json={"themes": ["psykologi", "helse"], "department": "Psykologi"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"], "Forventer minst ett treff"
    top = data["results"][0]
    assert top["citations"]


@pytest.mark.asyncio
async def test_match_endpoint_uses_precomputed_summary_language(client) -> None:
    response_no = await client.post(
        "/match",
        json={"themes": ["psykologi"], "department": "Psykologi"},
        headers={"X-UI-Language": "no"},
    )
    assert response_no.status_code == 200
    payload_no = response_no.json()["results"][0]
    why_no = payload_no["why"]
    assert "forhåndsgenerert" in why_no.lower()
    assert payload_no.get("whyByLang", {}).get("no")

    response_en = await client.post(
        "/match",
        json={"themes": ["psykologi"], "department": "Psykologi"},
        headers={"X-UI-Language": "en"},
    )
    assert response_en.status_code == 200
    payload_en = response_en.json()["results"][0]
    why_en = payload_en["why"]
    assert "precomputed summary" in why_en.lower()
    assert payload_en.get("whyByLang", {}).get("en")


@pytest.mark.asyncio
async def test_match_endpoint_offline_snapshot(offline_client) -> None:
    response = await offline_client.post(
        "/match",
        json={
            "themes": [
                "psykologi",
                "klinisk",
                "traumebehandling",
                "kognitiv",
            ],
            "department": "Psykologi",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"], "Forventer treff basert på offline snapshot"
    names = {item["name"] for item in data["results"]}
    assert "Alex Gillespie" in names


@pytest.mark.asyncio
async def test_match_endpoint_prefers_rag_when_index_ready(
    client, tmp_path_factory
) -> None:
    index_dir = tmp_path_factory.mktemp("rag_vectors")
    store = LocalVectorStore(index_dir)
    embedder = DummyEmbeddingBackend(dimension=3)

    chunk = Chunk(
        staff_slug="test-forsker",
        chunk_id="test-forsker-0000",
        text="Psykologi og helse forskning ved ONH.",
        order=0,
        token_count=6,
        source_url="https://example.org/test-forsker",
        metadata={
            "name": "RAG Forsker",
            "department": "Psykologi",
            "profile_url": "https://example.org/test-forsker",
            "title": "Førsteamanuensis",
        },
    )
    store.add(embedder.embed([chunk.text]), [chunk])

    retriever = EmbeddingRetriever(
        vector_store=store, embedder=embedder, min_score=0.0, max_chunks_per_staff=2
    )
    client.app.state.embedding_retriever = retriever  # type: ignore[attr-defined]
    client.app.state.vector_index_ready = True  # type: ignore[attr-defined]

    response = await client.post(
        "/match",
        json={"themes": ["psykologi", "helse"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"], "RAG-søket skal levere treff"
    top = data["results"][0]
    assert top["name"] == "RAG Forsker"
    assert top["citations"][0]["snippet"].startswith("Psykologi og helse")


@pytest.mark.asyncio
async def test_staff_document_cache_reuse(tmp_path) -> None:
    cache_manager = CacheManager(
        directory=tmp_path / "cache", retention_days=1, enabled=True
    )
    fetch_utils = FetchUtils(allowlist_domains=["example.com"], max_kb_per_page=10)

    fetch_calls = {"count": 0}

    async def fake_fetch_page(self, client_http, url):  # type: ignore[override]
        fetch_calls["count"] += 1
        return {"url": url, "title": "Stub", "text": "Lang tekst for caching."}

    fetch_utils.fetch_page = types.MethodType(fake_fetch_page, fetch_utils)

    profile = StaffProfile(
        name="Cache Test",
        department="Helsefag",
        profile_url="https://example.com/profile",
        sources=["https://example.com/source"],
        tags=[],
    )

    docs_first = await routes._fetch_staff_documents(  # type: ignore[attr-defined]
        staff_profiles=[profile],
        fetch_utils=fetch_utils,
        cache_manager=cache_manager,
        max_pages=1,
    )
    assert fetch_calls["count"] == 1
    assert docs_first and docs_first[0].combined_text

    docs_second = await routes._fetch_staff_documents(  # type: ignore[attr-defined]
        staff_profiles=[profile],
        fetch_utils=fetch_utils,
        cache_manager=cache_manager,
        max_pages=1,
    )
    assert fetch_calls["count"] == 1, "Expected cached StaffDocument to skip fetch"
    assert docs_second and docs_second[0].combined_text == docs_first[0].combined_text


@pytest.mark.asyncio
async def test_staff_fetch_cache_normalizes_bs4_strings(tmp_path) -> None:
    cache_manager = CacheManager(
        directory=tmp_path / "cache", retention_days=1, enabled=True
    )
    fetch_utils = FetchUtils(allowlist_domains=["example.com"], max_kb_per_page=10)

    async def fake_fetch_page(self, client_http, url):  # type: ignore[override]
        soup = BeautifulSoup("<html><title>Stub title</title></html>", "html.parser")
        return {
            "url": url,
            "title": soup.title.string,
            "text": "Lang tekst for caching.",
        }

    fetch_utils.fetch_page = types.MethodType(fake_fetch_page, fetch_utils)

    profile = StaffProfile(
        name="Cache Test",
        department="Helsefag",
        profile_url="https://example.com/profile",
        sources=["https://example.com/source"],
        tags=[],
    )

    docs = await routes._fetch_staff_documents(  # type: ignore[attr-defined]
        staff_profiles=[profile],
        fetch_utils=fetch_utils,
        cache_manager=cache_manager,
        max_pages=1,
    )
    assert docs

    cached = cache_manager.get("fetch::https://example.com/source")
    assert cached is not None
    assert cached["title"] == "Stub title"
    assert cached["title"].__class__ is str


def test_build_normalized_preview_prefers_sentences() -> None:
    text = (
        "Første setning beskriver psykologi. Andre setning utdyper helseperspektivet. "
        "Tredje setning er overflødig."
    )
    preview = routes._build_normalized_preview(text, limit=80)
    assert preview.endswith(".")
    assert "overflødig" not in preview


def test_chunks_to_citations_maps_nva_url() -> None:
    chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-0001",
        text="Resultatet beskriver digital sikkerhet og personvern i undervisning.",
        order=0,
        token_count=12,
        source_url="https://api.nva.unit.no/publication/abc123",
        metadata={
            "name": "Test Forsker",
            "nva_publication_id": "abc123",
        },
    )
    citations = routes._chunks_to_citations([chunk], themes=[])
    assert citations
    assert citations[0].url == "https://nva.sikt.no/registration/abc123"


def test_chunks_to_citations_prefers_nva_when_overlap_exists() -> None:
    staffinfo = Chunk(
        staff_slug="slug",
        chunk_id="slug-staffinfo-0000",
        text="Generell beskrivelse uten relevante nøkkelord.",
        order=0,
        token_count=6,
        source_url="staffinfo://slug",
        metadata={"source_kind": "staffinfo", "source_title": "Staffinfo"},
    )
    nva = Chunk(
        staff_slug="slug",
        chunk_id="slug-nva-0000",
        text="Denne artikkelen handler om personvern og digital sikkerhet.",
        order=1,
        token_count=9,
        source_url="https://api.nva.unit.no/publication/abc123",
        metadata={
            "source_kind": "nva",
            "source_title": "NVA",
            "nva_publication_id": "abc123",
        },
    )

    citations = routes._chunks_to_citations(
        [staffinfo, nva],
        themes=["personvern"],
        source_priority=["nva", "profile", "staffinfo"],
        min_query_overlap=1,
    )

    assert citations
    assert citations[0].url == "https://nva.sikt.no/registration/abc123"


def test_chunks_to_citations_fallbacks_when_nva_overlap_missing() -> None:
    nva = Chunk(
        staff_slug="slug",
        chunk_id="slug-nva-0000",
        text="Denne artikkelen handler om helt andre temaer.",
        order=0,
        token_count=8,
        source_url="https://api.nva.unit.no/publication/abc123",
        metadata={
            "source_kind": "nva",
            "source_title": "NVA",
            "nva_publication_id": "abc123",
        },
    )
    profile = Chunk(
        staff_slug="slug",
        chunk_id="slug-profile-0000",
        text="Profilen omtaler personvern i undervisning.",
        order=1,
        token_count=6,
        source_url="https://oslonyehoyskole.no/ansatt/test",
        metadata={"source_kind": "profile", "source_title": "Profil"},
    )

    citations = routes._chunks_to_citations(
        [nva, profile],
        themes=["personvern"],
        source_priority=["nva", "profile", "staffinfo"],
        min_query_overlap=1,
    )

    assert citations
    assert citations[0].url == "https://oslonyehoyskole.no/ansatt/test"


def test_chunks_to_citations_requires_stronger_profile_staffinfo_overlap() -> None:
    nva = Chunk(
        staff_slug="slug",
        chunk_id="slug-nva-0002",
        text="Denne publikasjonen handler om personvern.",
        order=0,
        token_count=6,
        source_url="https://api.nva.unit.no/publication/abc126",
        metadata={
            "source_kind": "nva",
            "source_title": "NVA",
            "nva_publication_id": "abc126",
        },
    )
    profile = Chunk(
        staff_slug="slug",
        chunk_id="slug-profile-0002",
        text="Profilen nevner personvern.",
        order=1,
        token_count=4,
        source_url="https://oslonyehoyskole.no/ansatt/test",
        metadata={"source_kind": "profile", "source_title": "Profil"},
    )

    citations = routes._chunks_to_citations(
        [profile, nva],
        themes=["personvern"],
        source_priority=["nva", "profile", "staffinfo"],
        min_query_overlap=1,
        profile_staffinfo_min_query_overlap=2,
    )

    assert citations
    assert citations[0].url == "https://nva.sikt.no/registration/abc126"


def test_chunks_to_citations_fallback_downranks_zero_overlap_sources() -> None:
    nva = Chunk(
        staff_slug="slug",
        chunk_id="slug-nva-0003",
        text="Publikasjonen dekker personvern i akademia.",
        order=0,
        token_count=6,
        source_url="https://api.nva.unit.no/publication/abc127",
        metadata={
            "source_kind": "nva",
            "source_title": "NVA",
            "nva_publication_id": "abc127",
        },
    )
    profile = Chunk(
        staff_slug="slug",
        chunk_id="slug-profile-0003",
        text="Profilen beskriver generell undervisningserfaring.",
        order=1,
        token_count=5,
        source_url="https://oslonyehoyskole.no/ansatt/test",
        metadata={"source_kind": "profile", "source_title": "Profil"},
    )

    citations = routes._chunks_to_citations(
        [profile, nva],
        themes=["personvern", "compliance"],
        source_priority=["nva", "profile", "staffinfo"],
        min_query_overlap=2,
        profile_staffinfo_min_query_overlap=3,
    )

    assert citations
    assert citations[0].url == "https://nva.sikt.no/registration/abc127"


def test_chunks_to_citations_publication_mode_uses_nva_tags_for_overlap() -> None:
    nva = Chunk(
        staff_slug="slug",
        chunk_id="slug-nva-0001",
        text="Predictors of internalising behaviour problems in adolescents.",
        order=0,
        token_count=10,
        source_url="https://api.nva.unit.no/publication/abc124",
        metadata={
            "source_kind": "nva",
            "source_title": "Predictors of internalising behaviour problems",
            "nva_publication_id": "abc124",
            "tags": ["Utviklingspsykopatologi", "Longitudinelle studier"],
        },
    )
    profile = Chunk(
        staff_slug="slug",
        chunk_id="slug-profile-0001",
        text="Profilen omtaler utviklingspsykopatologi i norske studier.",
        order=1,
        token_count=8,
        source_url="https://oslonyehoyskole.no/ansatt/test",
        metadata={"source_kind": "profile", "source_title": "Profil"},
    )

    publication_citations = routes._chunks_to_citations(
        [profile, nva],
        themes=["utviklingspsykopatologi"],
        source_priority=["nva", "profile"],
        min_query_overlap=1,
        match_mode="publication_grounded",
    )
    profile_citations = routes._chunks_to_citations(
        [profile, nva],
        themes=["utviklingspsykopatologi"],
        source_priority=["nva", "profile"],
        min_query_overlap=1,
        match_mode="profile_grounded",
    )

    assert publication_citations
    assert publication_citations[0].url == "https://nva.sikt.no/registration/abc124"
    assert "Nokkelord:" in publication_citations[0].snippet
    assert "Utviklingspsykopatologi" in publication_citations[0].snippet
    assert profile_citations
    assert profile_citations[0].url == "https://oslonyehoyskole.no/ansatt/test"


def test_keyword_overlap_score_uses_tokenized_themes() -> None:
    chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-0001",
        text="Digital sikkerhet og personvern i undervisning.",
        order=0,
        token_count=6,
        source_url="https://example.org/source",
        metadata={},
    )

    score = routes._keyword_overlap_score(  # type: ignore[attr-defined]
        [chunk], ["digital sikkerhet", "personvern"]
    )
    assert score == pytest.approx(1.0)


def test_method_overlap_score_detects_multiword_method_tokens() -> None:
    score = routes._method_overlap_score(  # type: ignore[attr-defined]
        ["Mixed methods", "Psykologi"],
        ["mixed methods i helsefag"],
    )
    assert score > 0.0


def test_tag_overlap_score_uses_concept_keyword_mapping() -> None:
    score = routes._tag_overlap_score(  # type: ignore[attr-defined]
        ["kvalitativ metode"],
        ["diskursanalyse"],
        concept_keyword_map={"diskursanalyse": ["kvalitativ metode"]},
    )
    assert score > 0.0


def test_synonym_expansion_handles_fn_and_humanitaerrett() -> None:
    expanded = routes._expand_themes_for_query(  # type: ignore[attr-defined]
        ["FN", "humanitærrett"],
        synonym_expansion_map={
            "fn": ["forente nasjoner", "un"],
            "humanitærrett": ["internasjonal humanitærrett", "ihl"],
        },
    )
    expanded_set = set(expanded)
    assert "fn" in expanded_set
    assert "forente" in expanded_set
    assert "nasjoner" in expanded_set
    assert "humanitærrett" in expanded_set
    assert "internasjonal" in expanded_set


def test_synonym_expansion_supports_diplomati_and_kat() -> None:
    score = routes._tag_overlap_score(  # type: ignore[attr-defined]
        ["kognitiv terapi", "diplomati"],
        ["kognitiv atferdsterapi", "utenrikspolitikk"],
        synonym_expansion_map={
            "kognitiv": ["kognitiv terapi"],
            "atferdsterapi": ["kognitiv terapi", "kat"],
            "utenrikspolitikk": ["diplomati"],
        },
    )
    assert score > 0.0


def test_score_breakdown_exposes_expanded_query_terms_count() -> None:
    chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-9999",
        text="Arbeid med internasjonal humanitærrett og konflikt.",
        order=0,
        token_count=7,
        source_url="https://example.org/source",
        metadata={
            "name": "Ekspansjon Test",
            "department": "Statsvitenskap",
            "profile_url": "https://example.org/profile",
            "tags": ["internasjonal humanitærrett"],
            "source_kind": "profile",
        },
    )
    result = RetrievalResult(
        staff_slug="slug",
        score=0.4,
        chunks=[chunk],
        staff_name="Ekspansjon Test",
        metadata={"department": "Statsvitenskap", "semantic_score": 0.4},
    )
    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )

    item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["humanitærrett"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles={
            "publication_grounded": {},
            "profile_grounded": {},
        },
        synonym_expansion_map={"humanitærrett": ["internasjonal humanitærrett", "ihl"]},
        overexposure_penalty_config={},
    )

    assert item is not None
    assert item.score_breakdown["expanded_query_terms_count"] > 0


def test_retrieval_result_scoring_weights_are_applied() -> None:
    chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-0002",
        text="Personvern og digital sikkerhet i undervisning.",
        order=0,
        token_count=6,
        source_url="https://example.org/source",
        metadata={
            "name": "Test Forsker",
            "department": "Psykologi",
            "profile_url": "https://example.org/profile",
            "tags": ["digital sikkerhet"],
            "source_kind": "profile",
        },
    )
    result = RetrievalResult(
        staff_slug="slug",
        score=0.4,
        chunks=[chunk],
        staff_name="Test Forsker",
        metadata={"department": "Psykologi"},
    )
    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )

    semantic_only = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["digital sikkerhet"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles={
            "publication_grounded": {},
            "profile_grounded": {},
        },
        overexposure_penalty_config={},
    )
    boosted = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["digital sikkerhet"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.2, "tags": 0.2, "methods": 0.0},
        mode_scoring_profiles={
            "publication_grounded": {},
            "profile_grounded": {},
        },
        overexposure_penalty_config={},
    )

    assert semantic_only is not None
    assert boosted is not None
    assert semantic_only.score == pytest.approx(0.4, abs=1e-4)
    assert boosted.score > semantic_only.score


def test_exact_keyword_promotion_ranks_exact_match_above_non_exact() -> None:
    exact_chunk = Chunk(
        staff_slug="slug-exact",
        chunk_id="slug-exact-0001",
        text="Fagprofil med kommunikasjonstrening i organisasjoner.",
        order=0,
        token_count=6,
        source_url="https://example.org/exact",
        metadata={
            "name": "Eksakt Match",
            "department": "Psykologi",
            "profile_url": "https://example.org/exact",
            "tags": ["kommunikasjonstrening"],
            "source_kind": "profile",
        },
    )
    non_exact_chunk = Chunk(
        staff_slug="slug-non-exact",
        chunk_id="slug-non-exact-0001",
        text="Bred organisasjonspsykologi og ledelse.",
        order=0,
        token_count=5,
        source_url="https://example.org/non-exact",
        metadata={
            "name": "Sterk Semantikk",
            "department": "Psykologi",
            "profile_url": "https://example.org/non-exact",
            "tags": ["organisasjon"],
            "source_kind": "profile",
        },
    )

    exact_result = RetrievalResult(
        staff_slug="slug-exact",
        score=0.35,
        chunks=[exact_chunk],
        staff_name="Eksakt Match",
        metadata={"department": "Psykologi", "semantic_score": 0.35},
    )
    non_exact_result = RetrievalResult(
        staff_slug="slug-non-exact",
        score=0.92,
        chunks=[non_exact_chunk],
        staff_name="Sterk Semantikk",
        metadata={"department": "Psykologi", "semantic_score": 0.92},
    )

    class StubRetriever:
        def __init__(self, results):
            self._results = results

        def retrieve(self, query):
            return self._results

    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )

    items = routes._match_via_retriever(  # type: ignore[attr-defined]
        retriever=StubRetriever([non_exact_result, exact_result]),
        payload=routes.MatchRequest(themes=["kommunikasjonstrening"], mode="profile_grounded"),
        match_mode="profile_grounded",
        max_candidates=10,
        translator=NoOpTranslator(),
        language_ctx=language_ctx,
        query_text="kommunikasjonstrening",
        citation_source_priority=["profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles={"publication_grounded": {}, "profile_grounded": {}},
        exact_keyword_promotion_config={"enabled": True, "keyword_bonus": 0.0, "min_token_length": 2},
        overexposure_penalty_config={"enabled": False},
    )

    assert len(items) == 2
    assert items[0].name == "Eksakt Match"
    assert items[0].score_breakdown["exact_keyword_match"] == pytest.approx(1.0)
    assert items[1].score_breakdown["exact_keyword_match"] == pytest.approx(0.0)


def test_exact_keyword_promotion_tie_keeps_score_order() -> None:
    high_score_exact_chunk = Chunk(
        staff_slug="slug-high",
        chunk_id="slug-high-0001",
        text="Kommunikasjonstrening med dokumentert effekt.",
        order=0,
        token_count=6,
        source_url="https://example.org/high",
        metadata={
            "name": "Eksakt Høy",
            "department": "Psykologi",
            "profile_url": "https://example.org/high",
            "tags": ["kommunikasjonstrening"],
            "source_kind": "profile",
        },
    )
    low_score_exact_chunk = Chunk(
        staff_slug="slug-low",
        chunk_id="slug-low-0001",
        text="Kommunikasjonstrening i praksis.",
        order=0,
        token_count=4,
        source_url="https://example.org/low",
        metadata={
            "name": "Eksakt Lav",
            "department": "Psykologi",
            "profile_url": "https://example.org/low",
            "tags": ["kommunikasjonstrening"],
            "source_kind": "profile",
        },
    )

    high_result = RetrievalResult(
        staff_slug="slug-high",
        score=0.8,
        chunks=[high_score_exact_chunk],
        staff_name="Eksakt Høy",
        metadata={"department": "Psykologi", "semantic_score": 0.8},
    )
    low_result = RetrievalResult(
        staff_slug="slug-low",
        score=0.4,
        chunks=[low_score_exact_chunk],
        staff_name="Eksakt Lav",
        metadata={"department": "Psykologi", "semantic_score": 0.4},
    )

    class StubRetriever:
        def __init__(self, results):
            self._results = results

        def retrieve(self, query):
            return self._results

    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )

    items = routes._match_via_retriever(  # type: ignore[attr-defined]
        retriever=StubRetriever([low_result, high_result]),
        payload=routes.MatchRequest(themes=["kommunikasjonstrening"], mode="profile_grounded"),
        match_mode="profile_grounded",
        max_candidates=10,
        translator=NoOpTranslator(),
        language_ctx=language_ctx,
        query_text="kommunikasjonstrening",
        citation_source_priority=["profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles={"publication_grounded": {}, "profile_grounded": {}},
        exact_keyword_promotion_config={"enabled": True, "keyword_bonus": 0.0, "min_token_length": 2},
        overexposure_penalty_config={"enabled": False},
    )

    assert len(items) == 2
    assert items[0].name == "Eksakt Høy"
    assert items[1].name == "Eksakt Lav"


def test_mode_scoring_prefers_nva_for_publication_grounded() -> None:
    nva_chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-0003",
        text="Personvern og digital sikkerhet i publisering.",
        order=0,
        token_count=7,
        source_url="https://api.nva.unit.no/publication/abc123",
        metadata={
            "name": "Publikasjonsforsker",
            "department": "Psykologi",
            "profile_url": "https://example.org/publikasjon",
            "source_kind": "nva",
            "nva_publication_id": "abc123",
        },
    )
    result = RetrievalResult(
        staff_slug="slug",
        score=0.4,
        chunks=[nva_chunk],
        staff_name="Publikasjonsforsker",
        metadata={"department": "Psykologi", "semantic_score": 0.4},
    )
    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )
    mode_profiles = {
        "publication_grounded": {
            "source_kind_boosts": {"nva": 0.2},
            "citation_overlap_weight": 0.1,
            "nva_citation_bonus": 0.1,
        },
        "profile_grounded": {
            "source_kind_boosts": {"profile": 0.2, "staffinfo": 0.1},
            "citation_overlap_weight": 0.0,
            "nva_citation_bonus": 0.0,
        },
    }

    publication_item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["personvern"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["nva", "profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles=mode_profiles,
        overexposure_penalty_config={},
    )
    profile_item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["personvern"],
        match_mode="profile_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["nva", "profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles=mode_profiles,
        overexposure_penalty_config={},
    )

    assert publication_item is not None
    assert profile_item is not None
    assert publication_item.score > profile_item.score
    assert publication_item.score_breakdown["mode_bonus"] > profile_item.score_breakdown["mode_bonus"]


def test_mode_scoring_prefers_profile_for_profile_grounded() -> None:
    profile_chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-0004",
        text="Klinisk psykologi og praksisnaer veiledning.",
        order=0,
        token_count=6,
        source_url="https://oslonyehoyskole.no/ansatt/test",
        metadata={
            "name": "Profilforsker",
            "department": "Psykologi",
            "profile_url": "https://oslonyehoyskole.no/ansatt/test",
            "source_kind": "profile",
        },
    )
    result = RetrievalResult(
        staff_slug="slug",
        score=0.4,
        chunks=[profile_chunk],
        staff_name="Profilforsker",
        metadata={"department": "Psykologi", "semantic_score": 0.4},
    )
    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )
    mode_profiles = {
        "publication_grounded": {
            "source_kind_boosts": {"nva": 0.2},
            "citation_overlap_weight": 0.1,
            "nva_citation_bonus": 0.1,
        },
        "profile_grounded": {
            "source_kind_boosts": {"profile": 0.25, "staffinfo": 0.1},
            "citation_overlap_weight": 0.0,
            "nva_citation_bonus": 0.0,
        },
    }

    publication_item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["klinisk psykologi"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["profile", "nva"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles=mode_profiles,
        overexposure_penalty_config={},
    )
    profile_item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["klinisk psykologi"],
        match_mode="profile_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["profile", "nva"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles=mode_profiles,
        overexposure_penalty_config={},
    )

    assert publication_item is not None
    assert profile_item is not None
    assert profile_item.score > publication_item.score
    assert profile_item.score_breakdown["mode_bonus"] > publication_item.score_breakdown["mode_bonus"]


def test_overexposure_penalty_reduces_low_signal_profile_spillover() -> None:
    profile_chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-0005",
        text="Bred erfaring med undervisning, ledelse og veiledning i organisasjoner.",
        order=0,
        token_count=9,
        source_url="https://oslonyehoyskole.no/ansatt/test",
        metadata={
            "name": "Bred Profil",
            "department": "Psykologi",
            "profile_url": "https://oslonyehoyskole.no/ansatt/test",
            "source_kind": "profile",
        },
    )
    result = RetrievalResult(
        staff_slug="slug",
        score=0.6,
        chunks=[profile_chunk],
        staff_name="Bred Profil",
        metadata={"department": "Psykologi", "semantic_score": 0.6},
    )
    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )

    item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["konfliktforskning", "etiopia"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["profile", "nva"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles={
            "publication_grounded": {},
            "profile_grounded": {},
        },
        overexposure_penalty_config={
            "enabled": True,
            "low_signal_threshold": 0.4,
            "profile_source_weight": 0.4,
            "staffinfo_source_weight": 0.3,
            "publication_without_nva_penalty": 0.08,
            "max_penalty": 0.2,
        },
    )

    assert item is not None
    assert item.score < 0.6
    assert item.score_breakdown["overexposure_penalty"] > 0.0


def test_category_intent_penalty_downweights_unrelated_department() -> None:
    profile_chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-cat-0001",
        text="Generell undervisningserfaring og administrasjon.",
        order=0,
        token_count=5,
        source_url="https://example.org/profile",
        metadata={
            "name": "Urelatert Profil",
            "department": "Psykologi",
            "profile_url": "https://example.org/profile",
            "source_kind": "profile",
            "tags": ["administrasjon"],
        },
    )
    result = RetrievalResult(
        staff_slug="slug",
        score=0.6,
        chunks=[profile_chunk],
        staff_name="Urelatert Profil",
        metadata={"department": "Psykologi", "semantic_score": 0.6},
    )
    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )

    item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["humanitærrett", "diplomati"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles={"publication_grounded": {}, "profile_grounded": {}},
        category_intent_penalty_config={
            "enabled": True,
            "base_penalty": 0.1,
            "max_penalty": 0.2,
            "evidence_overlap_threshold": 0.2,
            "intent_signals": {"international_relations": ["humanitærrett", "diplomati"]},
            "intent_department_map": {"international_relations": ["statsvitenskap"]},
        },
        overexposure_penalty_config={"enabled": False},
    )

    assert item is not None
    assert item.score_breakdown["category_intent_penalty"] > 0.0
    assert item.score < 0.6


def test_category_intent_penalty_keeps_cross_disciplinary_evidence() -> None:
    profile_chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-cat-0002",
        text="Forskning på humanitærrett og diplomati i konfliktsoner.",
        order=0,
        token_count=8,
        source_url="https://example.org/profile",
        metadata={
            "name": "Tverrfaglig Profil",
            "department": "Psykologi",
            "profile_url": "https://example.org/profile2",
            "source_kind": "profile",
            "tags": ["humanitærrett", "diplomati"],
        },
    )
    result = RetrievalResult(
        staff_slug="slug",
        score=0.6,
        chunks=[profile_chunk],
        staff_name="Tverrfaglig Profil",
        metadata={"department": "Psykologi", "semantic_score": 0.6},
    )
    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )

    item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["humanitærrett", "diplomati"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["profile"],
        min_query_overlap_per_citation=0,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles={"publication_grounded": {}, "profile_grounded": {}},
        category_intent_penalty_config={
            "enabled": True,
            "base_penalty": 0.1,
            "max_penalty": 0.2,
            "evidence_overlap_threshold": 0.2,
            "intent_signals": {"international_relations": ["humanitærrett", "diplomati"]},
            "intent_department_map": {"international_relations": ["statsvitenskap"]},
        },
        overexposure_penalty_config={"enabled": False},
    )

    assert item is not None
    assert item.score_breakdown["category_intent_penalty"] == pytest.approx(0.0, abs=1e-4)
    assert item.score == pytest.approx(0.6, abs=1e-4)


def test_overexposure_penalty_keeps_high_signal_nva_match_intact() -> None:
    nva_chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-0006",
        text="Konfliktforskning i Etiopia med analyser av fredsforhandlinger.",
        order=0,
        token_count=8,
        source_url="https://api.nva.unit.no/publication/abc333",
        metadata={
            "name": "Konfliktforsker",
            "department": "Statsvitenskap",
            "profile_url": "https://example.org/konflikt",
            "source_kind": "nva",
            "nva_publication_id": "abc333",
            "tags": ["konfliktforskning", "Etiopia"],
        },
    )
    result = RetrievalResult(
        staff_slug="slug",
        score=0.6,
        chunks=[nva_chunk],
        staff_name="Konfliktforsker",
        metadata={"department": "Statsvitenskap", "semantic_score": 0.6},
    )
    language_ctx = LanguageContext(
        user_lang="no",
        query_lang="no",
        embed_lang="no",
        llm_lang="no",
        translation_enabled=False,
        translate_for_embedding=False,
        translate_for_llm_input=False,
        translate_llm_output=False,
    )

    item = routes._retrieval_result_to_match_item(  # type: ignore[attr-defined]
        result,
        ["konfliktforskning", "etiopia"],
        match_mode="publication_grounded",
        language_ctx=language_ctx,
        citation_source_priority=["nva", "profile"],
        min_query_overlap_per_citation=1,
        scoring_weights={"semantic": 1.0, "keywords": 0.0, "tags": 0.0, "methods": 0.0},
        mode_scoring_profiles={
            "publication_grounded": {},
            "profile_grounded": {},
        },
        overexposure_penalty_config={
            "enabled": True,
            "low_signal_threshold": 0.4,
            "profile_source_weight": 0.4,
            "staffinfo_source_weight": 0.3,
            "publication_without_nva_penalty": 0.08,
            "max_penalty": 0.2,
        },
    )

    assert item is not None
    assert item.score == pytest.approx(0.6, abs=1e-4)
    assert item.score_breakdown["overexposure_penalty"] == pytest.approx(0.0, abs=1e-4)
