import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "sample_docs"


def _ingest_samples(service):
    for path in sorted(DOCS.glob("*.md")):
        service.ingest(path.name, path.read_text(encoding="utf-8"))


def test_retrieval_finds_the_right_document(service):
    _ingest_samples(service)
    results = service.retrieve("Quel est le plafond de remboursement pour un dîner ?", k=3)
    assert results[0].doc_name == "notes_de_frais.md"


def test_answer_cites_its_sources(service):
    _ingest_samples(service)
    result = service.ask("Combien de caractères doit contenir un mot de passe ?")
    assert "12 caractères" in result.answer
    assert re.search(r"\[\d\]", result.answer)  # la réponse cite au moins une source
    assert result.sources


def test_answer_without_documents(service):
    result = service.ask("Quelle est la politique de télétravail ?")
    assert "aucune information" in result.answer.lower()
