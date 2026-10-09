export type Source = {
  doc_id: string;
  doc_name: string;
  chunk_index: number;
  content: string;
  score: number;
};

export type AskResponse = {
  answer: string;
  sources: Source[];
  provider: string;
  latency_ms: number;
};

export type DocumentInfo = { doc_id: string; doc_name: string; chunks: number };

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (API_KEY) headers.set("X-API-Key", API_KEY);
  const response = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Erreur ${response.status}`);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export const api = {
  listDocuments: () => request<DocumentInfo[]>("/documents"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentInfo>("/documents/upload", { method: "POST", body: form });
  },
  remove: (docId: string) => request<void>(`/documents/${docId}`, { method: "DELETE" }),
  ask: (question: string) =>
    request<AskResponse>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
};
