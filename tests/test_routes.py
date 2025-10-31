from __future__ import annotations

from pathlib import Path


def test_analyze_topic_returns_themes(client) -> None:
    response = client.post(
        "/analyze-topic",
        data={"text": "Psykologi og helse ved Oslo Nye Høyskole."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["themes"]
    assert "normalizedPreview" in data


def test_match_endpoint_returns_ranked_result(client) -> None:
    response = client.post(
        "/match",
        json={"themes": ["psykologi", "helse"], "department": "Psykologi"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"], "Forventer minst ett treff"
    top = data["results"][0]
    assert top["citations"]


def test_shortlist_persistence_and_export(client, tmp_path) -> None:
    shortlist_payload = {
        "items": [
            {
                "id": "abc123",
                "name": "Test Forsker",
                "department": "Psykologi",
                "why": "Matcher psykologi.",
                "citations": [
                    {
                        "id": "[1]",
                        "title": "Stub",
                        "url": "https://example.com",
                        "snippet": "Forskning på psykologi.",
                    }
                ],
                "notes": "Prioritert kandidat",
            }
        ]
    }

    save_response = client.post("/shortlist", json=shortlist_payload)
    assert save_response.status_code == 200

    fetch_response = client.get("/shortlist")
    assert fetch_response.status_code == 200
    assert fetch_response.json()["items"][0]["name"] == "Test Forsker"

    export_response = client.post(
        "/shortlist/export",
        json={"format": "pdf", "metadata": {"topic": "Psykologi"}},
    )
    assert export_response.status_code == 200
    path_str = export_response.json()["path"]
    exporter_dir = client.app.state.shortlist_exporter.base_dir  # type: ignore[attr-defined]
    export_path = (Path(exporter_dir.parent.parent) / path_str).resolve()
    assert export_path.exists()


def test_match_endpoint_offline_snapshot(offline_client) -> None:
    response = offline_client.post(
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
