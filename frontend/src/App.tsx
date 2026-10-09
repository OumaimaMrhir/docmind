import { FormEvent, useEffect, useState } from "react";
import { api, AskResponse, DocumentInfo } from "./api";

type Turn = { question: string; response?: AskResponse; error?: string };

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");

  const refresh = () => api.listDocuments().then(setDocuments).catch((e) => setStatus(e.message));

  useEffect(() => {
    refresh();
  }, []);

  async function onUpload(files: FileList | null) {
    if (!files?.length) return;
    for (const file of Array.from(files)) {
      try {
        const doc = await api.upload(file);
        setStatus(`« ${doc.doc_name} » indexé (${doc.chunks} extraits).`);
      } catch (e) {
        setStatus(`Échec pour « ${file.name} » : ${(e as Error).message}`);
      }
    }
    refresh();
  }

  async function onAsk(event: FormEvent) {
    event.preventDefault();
    const q = question.trim();
    if (q.length < 3 || loading) return;
    setQuestion("");
    setLoading(true);
    setTurns((t) => [...t, { question: q }]);
    try {
      const response = await api.ask(q);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, response } : turn)));
    } catch (e) {
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error: (e as Error).message } : turn)));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>DocMind</h1>
        <p className="muted">Assistant IA sur votre base de connaissances interne.</p>

        <label className="upload">
          <input type="file" accept=".md,.txt,.pdf" multiple onChange={(e) => onUpload(e.target.files)} />
          + Ajouter des documents
        </label>
        {status && <p className="status">{status}</p>}

        <h2>Documents ({documents.length})</h2>
        <ul className="docs">
          {documents.map((d) => (
            <li key={d.doc_id}>
              <span title={d.doc_name}>{d.doc_name}</span>
              <small>{d.chunks} extraits</small>
              <button aria-label={`Supprimer ${d.doc_name}`} onClick={() => api.remove(d.doc_id).then(refresh)}>
                ×
              </button>
            </li>
          ))}
          {!documents.length && <li className="muted">Aucun document pour l'instant.</li>}
        </ul>
      </aside>

      <main className="chat">
        <div className="turns">
          {!turns.length && (
            <div className="empty">
              <h2>Posez une question</h2>
              <p className="muted">Exemple : « Combien de jours de télétravail sont autorisés par semaine ? »</p>
            </div>
          )}
          {turns.map((turn, i) => (
            <div key={i} className="turn">
              <div className="bubble user">{turn.question}</div>
              {turn.error && <div className="bubble error">{turn.error}</div>}
              {!turn.response && !turn.error && <div className="bubble bot muted">Recherche en cours…</div>}
              {turn.response && (
                <div className="bubble bot">
                  <p>{turn.response.answer}</p>
                  <details>
                    <summary>
                      {turn.response.sources.length} source{turn.response.sources.length > 1 ? "s" : ""} ·{" "}
                      {turn.response.provider} · {turn.response.latency_ms} ms
                    </summary>
                    <ol>
                      {turn.response.sources.map((s, j) => (
                        <li key={j}>
                          <strong>{s.doc_name}</strong> <small>score {s.score.toFixed(2)}</small>
                          <p>{s.content.length > 320 ? `${s.content.slice(0, 320)}…` : s.content}</p>
                        </li>
                      ))}
                    </ol>
                  </details>
                </div>
              )}
            </div>
          ))}
        </div>

        <form className="ask" onSubmit={onAsk}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Votre question…"
            aria-label="Question"
          />
          <button type="submit" disabled={loading || question.trim().length < 3}>
            Envoyer
          </button>
        </form>
      </main>
    </div>
  );
}
