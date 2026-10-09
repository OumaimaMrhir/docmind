"""Évaluation de la recherche : hit rate@k et MRR sur un jeu de questions annotées.

Usage : python -m eval.evaluate --docs ../sample_docs --k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.rag import build_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue la qualité de la recherche du RAG.")
    parser.add_argument("--docs", default="../sample_docs", help="Dossier des documents à indexer")
    parser.add_argument("--dataset", default=str(Path(__file__).with_name("dataset.jsonl")))
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    service = build_service(get_settings())
    for path in sorted(Path(args.docs).glob("*.md")):
        service.ingest(path.name, path.read_text(encoding="utf-8"))

    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    hits, reciprocal_ranks = 0, []
    for row in rows:
        results = service.retrieve(row["question"], args.k)
        names = [r.doc_name for r in results]
        rank = names.index(row["expected_doc"]) + 1 if row["expected_doc"] in names else None
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        status = f"rang {rank}" if rank else "MANQUÉ"
        print(f"[{status:>7}] {row['question']}")

    mrr = sum(reciprocal_ranks) / len(rows)
    print(f"\nHit rate@{args.k} : {hits / len(rows):.0%}  |  MRR : {mrr:.2f}  |  {len(rows)} questions")


if __name__ == "__main__":
    main()
