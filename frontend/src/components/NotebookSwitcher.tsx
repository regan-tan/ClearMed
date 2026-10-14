import { Check, ChevronDown, NotebookTabs, Plus, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { DEFAULT_TITLE, notebookStats, type NotebookMeta } from '../notebooks'

interface Props {
  notebooks: NotebookMeta[]
  activeId: string
  onRename: (title: string) => void
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: () => void
}

function plural(n: number, word: string) { return `${n} ${word}${n !== 1 ? 's' : ''}` }

// Notebook title (editable in place) plus a menu to switch, create and delete notebooks.
export function NotebookSwitcher({ notebooks, activeId, onRename, onSelect, onCreate, onDelete }: Props) {
  const active = notebooks.find(notebook => notebook.id === activeId)
  const [draft, setDraft] = useState(active?.title ?? '')
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)

  // Show the new title when switching notebooks.
  const [shownId, setShownId] = useState(activeId)
  if (shownId !== activeId) { setShownId(activeId); setDraft(active?.title ?? '') }

  useEffect(() => {
    if (!open) return
    function onPointer(e: MouseEvent) { if (!root.current?.contains(e.target as Node)) setOpen(false) }
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onPointer); document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onPointer); document.removeEventListener('keydown', onKey) }
  }, [open])

  function commit() {
    const title = draft.trim() || DEFAULT_TITLE
    setDraft(title)
    if (title !== active?.title) onRename(title)
  }

  const titleInput = (className: string) => <input className={className} aria-label="Notebook title" value={draft} onChange={e => setDraft(e.target.value)} onBlur={commit} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }} />

  return <div className="notebook-switcher" ref={root}>
    {titleInput('notebook-title')}
    <button className="icon-button switcher-toggle" onClick={() => setOpen(value => !value)} aria-expanded={open} aria-haspopup="true" aria-label="Switch notebook" title="Switch notebook"><NotebookTabs size={16} className="switcher-mobile-icon" /><ChevronDown size={15} /></button>
    {open && <div className="notebook-menu">
      {/* The header hides the title on phones, so rename lives in the menu there. */}
      <label className="notebook-menu-rename">Notebook title{titleInput('settings-select')}</label>
      <p className="notebook-menu-label">Your notebooks</p>
      <ul>{notebooks.map(notebook => {
        const stats = notebookStats(notebook.id)
        return <li key={notebook.id}><button className={notebook.id === activeId ? 'active' : ''} aria-current={notebook.id === activeId ? 'true' : undefined} onClick={() => { onSelect(notebook.id); setOpen(false) }}>
          <span className="min-w-0 flex-1"><span className="block truncate">{notebook.title}</span><small>{plural(stats.sources, 'source')} · {plural(stats.chats, 'chat')} · {plural(stats.cards, 'card')}</small></span>
          {notebook.id === activeId && <Check size={15} />}
        </button></li>
      })}</ul>
      <div className="notebook-menu-actions">
        <button onClick={() => { onCreate(); setOpen(false) }}><Plus size={15} />New notebook</button>
        <button className="danger-text" onClick={() => { onDelete(); setOpen(false) }}><Trash2 size={15} />Delete this notebook</button>
      </div>
    </div>}
  </div>
}
