import { Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { NotebookSource } from '../api/notebook'

interface Props {
  source: NotebookSource
  /** Flashcards generated from this source. */
  cardCount: number
  busy: boolean
  onConfirm: (deleteCards: boolean) => void
  onCancel: () => void
}

export function DeleteSourceDialog({ source, cardCount, busy, onConfirm, onCancel }: Props) {
  // Keep cards by default: they may hold review progress the student doesn't want to lose.
  const [deleteCards, setDeleteCards] = useState(false)

  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape' && !busy) onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onCancel])

  return <div className="dialog-backdrop" onMouseDown={e => { if (e.target === e.currentTarget && !busy) onCancel() }}>
    <div className="dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-source-title" aria-describedby="delete-source-body">
      <div className="dialog-icon"><Trash2 size={18} /></div>
      <h2 id="delete-source-title">Delete this source?</h2>
      <p id="delete-source-body"><strong>{source.name}</strong> will be removed from this notebook and won’t be used to answer questions.</p>
      {cardCount > 0 && <label className="dialog-option"><input type="checkbox" checked={deleteCards} onChange={e => setDeleteCards(e.target.checked)} />Also delete the {cardCount} flashcard{cardCount !== 1 ? 's' : ''} made from it</label>}
      <div className="dialog-actions">
        <button autoFocus onClick={onCancel} disabled={busy}>Cancel</button>
        <button className="danger" onClick={() => onConfirm(deleteCards)} disabled={busy}>{busy ? 'Deleting…' : 'Delete source'}</button>
      </div>
    </div>
  </div>
}
