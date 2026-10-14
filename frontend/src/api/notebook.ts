/** Proposed integration contract. Implement this adapter when the notebook backend is ready. */
import type { EmbeddingModel } from './types'

export interface NotebookSource {
  id: string
  name: string
  size: number
  status: 'staged' | 'processing' | 'ready' | 'error'
  error?: string
}
export interface Citation {
  source_id: string
  source_name: string
  excerpt: string
  page?: number
}
export interface NotebookMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}
export interface Conversation {
  id: string
  title: string
  messages: NotebookMessage[]
  updatedAt: number
}
export interface Note {
  id: string
  text: string
}
export interface NotebookRequest {
  message: string
  /** Empty when the user hasn't selected any sources; the backend answers without user material. */
  source_ids: string[]
  history: Pick<NotebookMessage, 'role' | 'content'>[]
  /** Extra study context for the answer, e.g. the flashcard being revised. */
  context?: string
  /** Search settings chosen in the header; shared with MCQ follow-ups. */
  retrieval: RetrievalSettings
}
export interface RetrievalSettings {
  embedding_model: EmbeddingModel
  top_k: number
}
export interface GeneratedFlashcard {
  front: string
  back: string
  source_id: string
  source_name: string
}
export interface NotebookSummary {
  id: string
  title: string
  createdAt: number
}
export interface NotebookApi {
  listNotebooks(): Promise<NotebookSummary[]>
  createNotebook(title: string): Promise<NotebookSummary>
  renameNotebook(notebookId: string, title: string): Promise<void>
  /** Deletes the notebook with its sources, chats, notes and flashcards. */
  deleteNotebook(notebookId: string): Promise<void>
  uploadSource(notebookId: string, file: File): Promise<NotebookSource>
  listSources(notebookId: string): Promise<NotebookSource[]>
  removeSource(notebookId: string, sourceId: string): Promise<void>
  chat(notebookId: string, request: NotebookRequest): Promise<{ content: string; citations: Citation[] }>
  /** Write question/answer cards from the given sources' content. */
  generateFlashcards(notebookId: string, request: { source_ids: string[] }): Promise<GeneratedFlashcard[]>
}

// UI-only adapter. No file bytes are read or transmitted, no extraction or retrieval occurs.
// Replace this object with your HTTP adapter; keep components independent of endpoint details.
// Notebooks themselves are kept in localStorage by the app (see src/notebooks.ts) until the backend owns them.
export const notebookApi: NotebookApi = {
  async listNotebooks() {
    return []
  },
  async createNotebook(title) {
    return { id: crypto.randomUUID(), title, createdAt: Date.now() }
  },
  async renameNotebook() {},
  async deleteNotebook() {},
  async uploadSource(_notebookId, file) {
    return { id: crypto.randomUUID(), name: file.name, size: file.size, status: 'staged' }
  },
  async listSources() {
    return []
  },
  async removeSource() {},
  async chat(_notebookId, { source_ids }) {
    return {
      content: source_ids.length
        ? 'Your question is ready for ClearMed. Once the backend is connected, this space will show an answer grounded in the sources you selected, with citations you can open to check the original passage. This is an interface preview, not an answer extracted from your files.'
        : 'Your question is ready for ClearMed. Once the backend is connected, this space will show an answer to your question. Add and select sources to ground answers in your own material. This is an interface preview, not a real answer.',
      citations: [],
    }
  },
  // Placeholder cards, clearly labelled: the preview can't read file contents to write real ones.
  async generateFlashcards(_notebookId, { source_ids }) {
    return source_ids.flatMap(source_id =>
      [1, 2, 3].map(n => ({
        source_id,
        source_name: '',
        front: `Preview card ${n}: a key concept from this source`,
        back: 'Once the backend is connected, ClearMed will write a question and answer here from the content of this source.',
      })),
    )
  },
}
