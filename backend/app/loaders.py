"""Lecture des fichiers déposés : texte, Markdown et PDF."""

from __future__ import annotations

import io

SUPPORTED = (".txt", ".md", ".pdf")


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Format non supporté. Formats acceptés : {', '.join(SUPPORTED)}")
