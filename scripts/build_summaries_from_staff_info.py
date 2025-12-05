"""Build precomputed staff summaries from staff_info.json.

Summaries are written to data/precomputed_summaries.json and loaded by the API
at startup. No runtime LLM calls are made during search.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "precomputed_summaries.json"
STAFF_INFO_PATH = PROJECT_ROOT / "staff_info.json"


PROMPT_TEMPLATE = """Du skriver en kort beskrivelse for en ansatt ved ONH basert på datafelt.
Skriv 55–80 ord på norsk, fordelt på 2–3 setninger, uten innledning eller punktlister.
Inkluder: (1) akademisk bakgrunn/rolle, (2) kurs/undervisning ved ONH hvis nevnt, (3) viktigste forsknings-/faglige interesser.
Bruk kun feltene nedenfor. Ikke legg til nye fakta. Ikke avslutt midt i en setning.
Felt:
Navn: {name}
Rolle: {role}
Kurs: {courses}
Forskning: {research}
Ekspertise: {expertise}
Annen: {other}
Oppsummering:"""


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _compose_deterministic(name: str, role: str, courses: list[str], research: list[str], expertise: list[str], other: list[str]) -> str:
    parts: list[str] = []
    if role:
        parts.append(f"{name} er {role}.")
    if courses:
        parts.append(f"Underviser i {', '.join(courses[:3])}.")
    interests_source = research or expertise or other
    if interests_source:
        parts.append(f"Arbeider med {', '.join(interests_source[:4])}.")
    text = " ".join(parts).strip()
    return truncate_words(text or f"{name} er ansatt ved ONH.", 80)


def truncate_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    truncated = " ".join(words[:max_words])
    last_period = truncated.rfind(".")
    if last_period > 40:
        return truncated[: last_period + 1].strip()
    return truncated.rstrip(",;:.") + "..."


async def generate_summary_llm(
    *,
    client: httpx.AsyncClient,
    model_name: str,
    endpoint: str,
    timeout: float,
    name: str,
    role: str,
    courses: list[str],
    research: list[str],
    expertise: list[str],
    other: list[str],
) -> str:
    prompt = PROMPT_TEMPLATE.format(
        name=name,
        role=role or "ukjent",
        courses=", ".join(courses) or "ikke oppgitt",
        research=", ".join(research) or "ikke oppgitt",
        expertise=", ".join(expertise) or "ikke oppgitt",
        other=", ".join(other) or "ikke oppgitt",
    )
    try:
        resp = await client.post(
            f"{endpoint}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        return truncate_words(text, 80)
    except Exception:
        return ""


async def main(use_llm: bool, selected_names: set[str] | None) -> None:
    if not STAFF_INFO_PATH.exists():
        raise FileNotFoundError(f"Missing staff_info.json at {STAFF_INFO_PATH}")
    staff_data = json.loads(STAFF_INFO_PATH.read_text(encoding="utf-8")).get("staff", [])
    total = len([s for s in staff_data if s.get("name") and (not selected_names or s.get("name") in selected_names)])

    # Load model config only if LLM is enabled.
    model_name = endpoint = None
    timeout = 120.0
    if use_llm:
        models_path = PROJECT_ROOT / "data" / "models.yaml"
        if not models_path.exists():
            raise FileNotFoundError("data/models.yaml not found for LLM config.")
        import yaml  # type: ignore

        models_cfg = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
        llm_cfg = models_cfg.get("llm_model", {})
        model_name = llm_cfg.get("name")
        endpoint = (llm_cfg.get("endpoint") or "http://localhost:11434").rstrip("/")
        timeout = float(llm_cfg.get("timeout") or 120)
        if not model_name:
            raise ValueError("llm_model.name is missing in data/models.yaml")

    summaries: Dict[str, str] = {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        processed = 0
        for entry in staff_data:
            name = entry.get("name", "").strip()
            if not name:
                continue
            if selected_names and name not in selected_names:
                continue
            role = entry.get("job_title", "").strip()
            courses = _normalize_list(entry.get("teaching_courses"))
            research = _normalize_list(entry.get("research_focus"))
            expertise = _normalize_list(entry.get("expertise_domains"))
            other = _normalize_list(entry.get("other_relevant_expertise"))

            summary = ""
            if use_llm and model_name and endpoint:
                summary = await generate_summary_llm(
                    client=client,
                    model_name=model_name,
                    endpoint=endpoint,
                    timeout=timeout,
                    name=name,
                    role=role,
                    courses=courses,
                    research=research,
                    expertise=expertise,
                    other=other,
                )
            if not summary:
                summary = _compose_deterministic(name, role, courses, research, expertise, other)

            summaries[name] = summary
            # If profile_url existed in staff_info, also key it. (Optional)
            profile_url = entry.get("profile_url")
            if profile_url:
                summaries[str(profile_url).strip()] = summary
            processed += 1
            print(f"{name} ferdig ({processed}/{total})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summaries": summaries,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(summaries)} summaries to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build staff summaries from staff_info.json")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM; use deterministic summaries only.")
    parser.add_argument("--staff", nargs="*", help="Optional list of staff names to regenerate.")
    args = parser.parse_args()
    selected = set(args.staff) if args.staff else None
    asyncio.run(main(use_llm=not args.no_llm, selected_names=selected))
