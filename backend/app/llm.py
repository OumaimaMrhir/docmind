"""Génération de la réponse : OpenAI, Anthropic, ou mode extractif sans clé API."""

from __future__ import annotations

import re
from typing import Protocol

import httpx

from .embeddings import keywords
from .store import SearchResult

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond aux questions des collaborateurs à partir de la base de "
    "connaissances interne. Réponds uniquement à partir des extraits fournis. Cite tes sources "
    "avec leur numéro entre crochets, par exemple [1]. Si les extraits ne contiennent pas la "
    "réponse, dis-le clairement au lieu d'inventer."
)


def build_prompt(question: str, contexts: list[SearchResult]) -> str:
    blocks = [f"[{i}] ({c.doc_name})\n{c.content}" for i, c in enumerate(contexts, start=1)]
    return "Extraits de la base de connaissances :\n\n" + "\n\n".join(blocks) + f"\n\nQuestion : {question}"


class LLM(Protocol):
    name: str

    def answer(self, question: str, contexts: list[SearchResult]) -> str: ...


def _keywords(text: str) -> set[str]:
    return set(keywords(text))


class ExtractiveLLM:
    """Sans clé API : renvoie les phrases des meilleurs extraits qui répondent le mieux à la question."""

    name = "extractive"

    def answer(self, question: str, contexts: list[SearchResult]) -> str:
        if not contexts:
            return "Je n'ai trouvé aucune information sur ce sujet dans la base de connaissances."
        q_words = _keywords(question)
        best_score = contexts[0].score
        candidates: list[tuple[float, float, int, str]] = []
        for i, ctx in enumerate(contexts, start=1):
            if ctx.score < 0.6 * best_score:  # on ne garde que les extraits proches du meilleur
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", ctx.content):
                words = _keywords(sentence)
                if not words or sentence.lstrip().startswith("#"):
                    continue
                overlap = len(q_words & words) / (len(q_words) or 1)
                candidates.append((overlap, ctx.score, i, sentence.strip()))
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        best = [c for c in candidates[:2] if c[0] >= 0.3]
        if not best:
            return "Je n'ai pas trouvé de réponse précise dans la base de connaissances."
        return " ".join(f"{sentence} [{i}]" for _, _, i, sentence in best)


class OpenAILLM:
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY est requise pour LLM_PROVIDER=openai")
        self.api_key, self.model = api_key, model

    def answer(self, question: str, contexts: list[SearchResult]) -> str:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, contexts)},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY est requise pour LLM_PROVIDER=anthropic")
        self.api_key, self.model = api_key, model

    def answer(self, question: str, contexts: list[SearchResult]) -> str:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 800,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": build_prompt(question, contexts)}],
            },
            timeout=60,
        )
        response.raise_for_status()
        parts = response.json()["content"]
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
