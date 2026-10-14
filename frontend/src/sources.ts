import type { NotebookSource } from './api/notebook'

/** Sources that can ground answers: ready on the server, or staged in the preview. */
export function isUsableSource(source: NotebookSource) {
  return source.status === 'ready' || source.status === 'staged'
}

export const SOURCE_STATUS_LABEL: Record<NotebookSource['status'], string> = {
  staged: 'Staged locally',
  processing: 'Processing…',
  ready: 'Ready',
  error: 'Couldn’t process',
}

// Client-side checks for uploads. These are conveniences; the backend must validate uploads itself.
export const ACCEPTED_SOURCE_TYPES = '.pdf,.docx,.txt,.md'
export const SOURCE_TYPES_LABEL = 'PDF, DOCX, TXT or MD, up to 20 MB each'
const MAX_SOURCE_BYTES = 20 * 1024 * 1024

/** Returns a message explaining why the file can't be added, or null if it's fine. */
export function sourceFileProblem(file: File) {
  if (!/\.(pdf|docx|txt|md)$/i.test(file.name)) return `${file.name}: unsupported file type.`
  if (!file.size || file.size > MAX_SOURCE_BYTES) return `${file.name}: choose a non-empty file under 20 MB.`
  return null
}
