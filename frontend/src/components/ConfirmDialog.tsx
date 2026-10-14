import { Trash2 } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'

interface Props {
  title: string
  children: ReactNode
  confirmLabel: string
  busy: boolean
  /** Optional extra choice shown as an unchecked checkbox, e.g. "Also delete its flashcards". */
  option?: string
  onConfirm: (optionChecked: boolean) => void
  onCancel: () => void
}

// Confirmation for destructive actions.
export function ConfirmDialog({ title, children, confirmLabel, busy, option, onConfirm, onCancel }: Props) {
  // Unchecked by default so extra data (e.g. review progress) isn't lost by accident.
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape' && !busy) onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onCancel])

  return <div className="dialog-backdrop" onMouseDown={e => { if (e.target === e.currentTarget && !busy) onCancel() }}>
    <div className="dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" aria-describedby="confirm-body">
      <div className="dialog-icon"><Trash2 size={18} /></div>
      <h2 id="confirm-title">{title}</h2>
      <p id="confirm-body">{children}</p>
      {option && <label className="dialog-option"><input type="checkbox" checked={checked} onChange={e => setChecked(e.target.checked)} />{option}</label>}
      <div className="dialog-actions">
        <button autoFocus onClick={onCancel} disabled={busy}>Cancel</button>
        <button className="danger" onClick={() => onConfirm(checked)} disabled={busy}>{busy ? 'Deleting…' : confirmLabel}</button>
      </div>
    </div>
  </div>
}
