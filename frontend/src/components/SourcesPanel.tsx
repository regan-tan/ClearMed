import { FileText, Plus, Trash2, Upload, X } from 'lucide-react'
import { useRef, useState } from 'react'
import type { NotebookSource } from '../api/notebook'

interface Props {
  sources: NotebookSource[]; selected: string[]; busy: boolean
  onAdd: (files: File[]) => void; onToggle: (id: string) => void
  onSelectAll: () => void; onRemove: (id: string) => void; onClose: () => void
}
export function SourcesPanel({ sources, selected, busy, onAdd, onToggle, onSelectAll, onRemove, onClose }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  return <aside className={`notebook-panel sources-panel ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false) }} onDrop={e => { e.preventDefault(); setDragging(false); if (!busy) onAdd(Array.from(e.dataTransfer.files)) }}>
    <div className="panel-heading"><h2>Sources <span className="count-badge">{sources.length}</span></h2><button className="icon-button mobile-close" aria-label="Close sources" onClick={onClose}><X size={17} /></button></div>
    <div className="source-controls"><input ref={input} type="file" multiple accept=".pdf,.docx,.txt,.md" className="hidden" onChange={e => { onAdd(Array.from(e.target.files ?? [])); e.target.value = '' }} /><button className="add-source" disabled={busy} onClick={() => input.current?.click()}><Plus size={17} />{busy ? 'Adding sources…' : 'Add sources'}</button><p>PDF, DOCX, TXT or MD · max 20 MB each</p></div>
    {sources.length ? <><label className="select-all"><input type="checkbox" checked={selected.length === sources.length} onChange={onSelectAll} />Select all sources</label><ul className="source-list">{sources.map(source => <li key={source.id}><input type="checkbox" checked={selected.includes(source.id)} onChange={() => onToggle(source.id)} aria-label={`Use ${source.name}`} /><FileText size={18} /><div className="min-w-0 flex-1"><p title={source.name}>{source.name}</p><small>{(source.size / 1024).toFixed(0)} KB · {source.status === 'staged' ? 'Staged locally' : source.status}</small>{source.error && <small>{source.error}</small>}</div><button className="icon-button delete-button" aria-label={`Delete ${source.name}`} title="Delete source" disabled={busy} onClick={() => onRemove(source.id)}><Trash2 size={15} /></button></li>)}</ul><p className="source-footnote">Selected sources will be included in your next question. Staged files haven’t been indexed.</p></> : <div className="source-empty"><div className="empty-icon"><Upload size={24} strokeWidth={1.4} /></div><h3>Your material, all together.</h3><p>Drop lecture notes, textbook chapters or revision guides here.</p><button onClick={() => input.current?.click()} className="text-link">Choose your first source</button></div>}
    <div className="local-notice"><span className="status-dot" />Preview mode · files stay on your device</div>
  </aside>
}
