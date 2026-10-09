def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_list_ask_delete(client):
    created = client.post("/documents/text", json={"name": "rh.md", "content": "Les congés payés sont de 25 jours par an."})
    assert created.status_code == 201
    doc_id = created.json()["doc_id"]

    docs = client.get("/documents").json()
    assert [d["doc_name"] for d in docs] == ["rh.md"]

    answer = client.post("/ask", json={"question": "Combien de jours de congés payés par an ?"})
    assert answer.status_code == 200
    body = answer.json()
    assert "25 jours" in body["answer"]
    assert body["sources"][0]["doc_name"] == "rh.md"

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == []


def test_upload_markdown_file(client):
    content = b"# Guide\n\nLe badge est remis le premier jour."
    response = client.post("/documents/upload", files={"file": ("guide.md", content, "text/markdown")})
    assert response.status_code == 201
    assert response.json()["chunks"] == 1


def test_unsupported_format_is_rejected(client):
    response = client.post("/documents/upload", files={"file": ("image.png", b"\x89PNG", "image/png")})
    assert response.status_code == 422


def test_question_too_short_is_rejected(client):
    assert client.post("/ask", json={"question": "a"}).status_code == 422


def test_api_key_is_enforced(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "secret"}).status_code == 200
