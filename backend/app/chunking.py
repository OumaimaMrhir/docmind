"""Découpage des documents en morceaux (chunks) avec chevauchement."""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Découpe un texte en chunks d'environ `chunk_size` caractères.

    Le découpage respecte d'abord les paragraphes, puis les phrases, pour ne pas
    couper une idée au milieu. Un chevauchement de `overlap` caractères garde
    le contexte entre deux chunks consécutifs.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être positif")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap doit être compris entre 0 et chunk_size")

    text = normalize(text)
    if not text:
        return []

    # Unités : paragraphes, eux-mêmes redécoupés en phrases s'ils sont trop longs
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            while len(sentence) > chunk_size:  # phrase anormalement longue
                units.append(sentence[:chunk_size])
                sentence = sentence[chunk_size - overlap :]
            if sentence:
                units.append(sentence)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail} {unit}".strip() if tail else unit
        if len(current) > chunk_size:  # le chevauchement ne doit pas dépasser la taille max
            current = unit
    if current:
        chunks.append(current)
    return chunks
