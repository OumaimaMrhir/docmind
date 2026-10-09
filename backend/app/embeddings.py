"""Embeddings : un modèle OpenAI, ou un embedding local par hachage (sans clé API)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

_WORD = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


STOPWORDS = set(
    "le la les un une des de du au aux et ou en dans sur pour par avec sans ce cet cette ces qui que quoi "
    "quel quelle quels quelles est sont être etre ont il elle ils elles on je tu nous vous mon ma mes ton ta tes son sa ses "
    "leur leurs ne pas plus se si faire fait dois doit peut puis combien comment quand the and of to is in".split()
)


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if len(w) > 1]


def keywords(text: str) -> list[str]:
    """Mots porteurs de sens, réduits à une racine simple (6 premières lettres)."""
    return [w[:6] for w in tokenize(text) if w not in STOPWORDS]


class HashingEmbedder:
    """Embedding local par « hashing trick » sur les mots et bigrammes.

    Moins précis qu'un modèle neuronal, mais déterministe, gratuit et sans
    dépendance : idéal pour les tests, la CI et une démo sans clé API.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _index(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            words = keywords(text)
            counts: dict[str, int] = {}
            for feature in words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]:
                counts[feature] = counts.get(feature, 0) + 1
            for feature, count in counts.items():
                idx, sign = self._index(feature)
                vec[idx] += sign * (1 + math.log(count))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY est requise pour EMBEDDING_PROVIDER=openai")
        self.api_key = api_key
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
