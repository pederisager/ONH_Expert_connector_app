from app.nva_lookup import preferred_nva_url


def test_preferred_url_uses_doi_when_available():
    pub_id = "0198ad0aec30-562681b0-e663-47be-9863-b7caa89c69e8"
    url = preferred_nva_url(pub_id)
    assert url.startswith("https://doi.org/10.1016/j.ejca.2013.02.029")


def test_preferred_url_fallbacks_to_api_when_no_doi():
    pub_id = "0198cc7ca9eb-dca3d983-f87b-4001-b098-ffded97558ba"
    url = preferred_nva_url(pub_id)
    assert url.startswith("https://api.")
