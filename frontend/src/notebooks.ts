import { hasStored, moveStored, readStored, removeStored, writeStored } from './storage'

export interface NotebookMeta { id: string; title: string; createdAt: number }

export const DEFAULT_TITLE = 'Untitled notebook'

// Data owned by each notebook. Practice progress, search settings and theme are shared.
const NOTEBOOK_DATA = ['sources', 'selected-sources', 'chats', 'active-chat', 'notes', 'flashcards'] as const
type NotebookData = typeof NOTEBOOK_DATA[number]

export function notebookKey(id: string, name: NotebookData) {
  return `notebook:${id}:${name}`
}

export function createNotebookMeta(title = DEFAULT_TITLE): NotebookMeta {
  return { id: crypto.randomUUID(), title, createdAt: Date.now() }
}

export function deleteNotebookData(id: string) {
  NOTEBOOK_DATA.forEach(name => removeStored(notebookKey(id, name)))
}

export function notebookStats(id: string) {
  const count = (name: NotebookData) => readStored<unknown[]>(notebookKey(id, name), []).length
  return { sources: count('sources'), chats: count('chats'), cards: count('flashcards') }
}

// Before notebooks existed, one notebook's data was stored under unscoped keys.
// Move it into a first notebook so nothing is lost, and make sure at least one notebook exists.
export function ensureNotebooks() {
  if (hasStored('notebooks')) return
  const notebook = createNotebookMeta(readStored('title', DEFAULT_TITLE))
  NOTEBOOK_DATA.forEach(name => moveStored(name, notebookKey(notebook.id, name)))
  removeStored('title')
  writeStored('notebooks', [notebook])
  writeStored('active-notebook', notebook.id)
}
