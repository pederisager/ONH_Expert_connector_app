"""Shortlist export helpers for JSON and PDF outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import orjson
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


class ShortlistExporter:
    """Writes shortlist exports to disk."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def export(self, format_: str, items: Iterable[Mapping], metadata: Mapping | None = None) -> Path:
        metadata = metadata or {}
        if format_ == "json":
            return self._export_json(items, metadata)
        if format_ == "pdf":
            return self._export_pdf(items, metadata)
        raise ValueError(f"Ukjent eksportformat: {format_}")

    def _timestamp(self) -> tuple[str, datetime]:
        now = datetime.now(timezone.utc)
        safe = now.strftime("%Y%m%d-%H%M%S")
        return safe, now

    def _export_json(self, items: Iterable[Mapping], metadata: Mapping) -> Path:
        safe, now = self._timestamp()
        payload = {
            "generated_at": now.isoformat(),
            "metadata": metadata,
            "items": list(items),
        }
        path = self.base_dir / f"shortlist-{safe}.json"
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        return path

    def _export_pdf(self, items: Iterable[Mapping], metadata: Mapping) -> Path:
        from textwrap import wrap

        safe, now = self._timestamp()
        path = self.base_dir / f"shortlist-{safe}.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4

        margin_x = 40
        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin_x, y, "ONH Expert Connector – Kortliste")
        y -= 24

        c.setFont("Helvetica", 10)
        c.drawString(margin_x, y, f"Generert: {now.astimezone().strftime('%d.%m.%Y %H:%M')}")
        y -= 16

        if metadata:
            meta_lines = []
            topic = metadata.get("topic")
            department = metadata.get("department")
            themes = metadata.get("themes") or []
            if topic:
                meta_lines.append(f"Tema: {topic[:200]}")
            if department:
                meta_lines.append(f"Avdeling: {department}")
            if themes:
                meta_lines.append("Temaer: " + ", ".join(themes))
            for line in meta_lines:
                for segment in wrap(line, width=90):
                    c.drawString(margin_x, y, segment)
                    y -= 14
            y -= 8

        for item in items:
            if y < 120:
                c.showPage()
                c.setFont("Helvetica-Bold", 16)
                y = height - 60

            name = item.get("name", "Ukjent")
            department = item.get("department", "")
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, y, f"{name} ({department})")
            y -= 16

            explanation = item.get("why") or "Ingen forklaring tilgjengelig."
            c.setFont("Helvetica", 10)
            for segment in wrap(explanation, width=95):
                c.drawString(margin_x, y, segment)
                y -= 14

            citations = item.get("citations") or []
            if citations:
                c.setFont("Helvetica-Oblique", 9)
                for citation in citations:
                    snippet = citation.get("snippet", "")
                    title = citation.get("title", "")
                    url = citation.get("url", "")
                    citation_line = f"{citation.get('id', '')} {title} – {url}"
                    for segment in wrap(citation_line, width=100):
                        c.drawString(margin_x, y, segment)
                        y -= 12
                    for segment in wrap(snippet, width=100):
                        c.drawString(margin_x + 12, y, segment)
                        y -= 12
            notes = item.get("notes")
            if notes:
                c.setFont("Helvetica", 9)
                c.drawString(margin_x, y, "Notater:")
                y -= 12
                for segment in wrap(notes, width=95):
                    c.drawString(margin_x + 12, y, segment)
                    y -= 12
            y -= 10

        c.save()
        return path
