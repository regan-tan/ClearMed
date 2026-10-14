/** Proposed integration contract. Implement this adapter when the notebook backend is ready. */
export interface NotebookSource {
  id: string
  name: string
  size: number
  status: 'staged' | 'processing' | 'ready' | 'error'
  error?: string
}
export interface Citation { source_id: string; source_name: string; excerpt: string; page?: number }
export interface NotebookMessage { id: string; role: 'user' | 'assistant'; content: string; citations?: Citation[] }
export interface Conversation { id: string; title: string; messages: NotebookMessage[]; updatedAt: number }
export interface Note { id: string; text: string }
export interface NotebookRequest {
  message: string
  /** Empty when the user hasn't selected any sources; the backend answers without user material. */
  source_ids: string[]
  history: Pick<NotebookMessage, 'role' | 'content'>[]
  /** Extra study context for the answer, e.g. the flashcard being revised. */
  context?: string
}
export interface GeneratedFlashcard { front: string; back: string; source_id: string; source_name: string }
export interface NotebookApi {
  uploadSource(file: File): Promise<NotebookSource>
  listSources(): Promise<NotebookSource[]>
  removeSource(id: string): Promise<void>
  chat(request: NotebookRequest): Promise<{ content: string; citations: Citation[] }>
  /** Write question/answer cards from the given sources' content. */
  generateFlashcards(request: { source_ids: string[] }): Promise<GeneratedFlashcard[]>
}

// UI-only adapter. No file bytes are read or transmitted, no extraction or retrieval occurs.
// Replace this object with your HTTP adapter; keep components independent of endpoint details.
export const notebookApi: NotebookApi = {
  async uploadSource(file) {
    return { id: crypto.randomUUID(), name: file.name, size: file.size, status: 'staged' }
  },
  async listSources() { return [] },
  async removeSource() {},
  async chat({ source_ids }) {
    return {
      content: source_ids.length
        ? 'Your question is ready for ClearMed. Once the backend is connected, this space will show an answer grounded in the sources you selected, with citations you can open to check the original passage. This is an interface preview, not an answer extracted from your files.'
        : 'Your question is ready for ClearMed. Once the backend is connected, this space will show an answer to your question. Add and select sources to ground answers in your own material. This is an interface preview, not a real answer.',
      citations: [],
    }
  },
  // Placeholder cards, clearly labelled: the preview can't read file contents to write real ones.
  async generateFlashcards({ source_ids }) {
    return source_ids.flatMap(source_id => [1, 2, 3].map(n => ({
      source_id,
      source_name: '',
      front: `Preview card ${n}: a key concept from this source`,
      back: 'Once the backend is connected, ClearMed will write a question and answer here from the content of this source.',
    })))
  },
}
