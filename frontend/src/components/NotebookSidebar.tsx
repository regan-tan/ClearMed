import { FileText, Loader2, Plus, SquarePen, Trash2, Upload, X } from 'lucide-react'
import { useRef, useState, type CSSProperties, type Dispatch, type KeyboardEvent, type PointerEvent, type SetStateAction } from 'react'
import type { Conversation, NotebookSource } from '../api/notebook'
import { ChatHistoryList } from './ChatHistoryList'
import { SidebarSection } from './SidebarSection'

export interface SidebarLayout {
  sourcesOpen: boolean
  chatsOpen: boolean
  /** Share of the height given to Sources when both sections are open (0–1). */
  split: number
}

interface Props {
  sources: NotebookSource[]; selected: string[]; busy: boolean
  onAdd: (files: File[]) => void; onToggle: (id: string) => void
  onSelectAll: () => void; onRemove: (id: string) => void; onClose: () => void
  chats: Conversation[]; activeChatId: string
  onSelectChat: (id: string) => void; onNewChat: () => void; onDeleteChat: (id: string) => void
  layout: SidebarLayout; onLayoutChange: Dispatch<SetStateAction<SidebarLayout>>
}

const clampSplit = (value: number) => Math.min(0.8, Math.max(0.2, value))

function sectionStyle(open: boolean, otherOpen: boolean, share: number): CSSProperties {
  if (!open) return { flex: '0 0 auto' }
  return { flex: otherOpen ? `${share} 1 0` : '1 1 0' }
}

// Left sidebar with stacked, collapsible Sources and Chats sections and a draggable divider between them.
export function NotebookSidebar(props: Props) {
  const { sources, selected, busy, onAdd, onToggle, onSelectAll, onRemove, onClose, chats, activeChatId, onSelectChat, onNewChat, onDeleteChat, layout, onLayoutChange } = props
  const input = useRef<HTMLInputElement>(null)
  const stack = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState(false)
  const bothOpen = layout.sourcesOpen && layout.chatsOpen

  function startResize(e: PointerEvent<HTMLDivElement>) {
    const box = stack.current?.getBoundingClientRect()
    if (!box) return
    e.preventDefault()
    const handle = e.currentTarget
    try { handle.setPointerCapture(e.pointerId) } catch { /* pointer already released */ }
    const move = (event: globalThis.PointerEvent) => onLayoutChange(current => ({ ...current, split: clampSplit((event.clientY - box.top) / box.height) }))
    const stop = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', stop); window.removeEventListener('pointercancel', stop) }
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', stop); window.addEventListener('pointercancel', stop)
  }
  function resizeWithKeys(e: KeyboardEvent<HTMLDivElement>) {
    const step = e.key === 'ArrowUp' ? -0.05 : e.key === 'ArrowDown' ? 0.05 : 0
    if (!step) return
    e.preventDefault()
    onLayoutChange(current => ({ ...current, split: clampSplit(current.split + step) }))
  }

  return <aside className="notebook-panel sidebar" aria-label="Sources and chats">
    <input ref={input} type="file" multiple accept=".pdf,.docx,.txt,.md" className="hidden" onChange={e => { onAdd(Array.from(e.target.files ?? [])); e.target.value = '' }} />
    <div className="sidebar-stack" ref={stack}>
      <SidebarSection title="Sources" count={sources.length} open={layout.sourcesOpen} onToggle={() => onLayoutChange(current => ({ ...current, sourcesOpen: !current.sourcesOpen }))}
        style={sectionStyle(layout.sourcesOpen, layout.chatsOpen, layout.split)}
        actions={<>
          <button className="icon-button" disabled={busy} onClick={() => input.current?.click()} aria-label="Add sources" title="Add sources (PDF, DOCX, TXT or MD · max 20 MB each)">{busy ? <Loader2 size={15} className="animate-spin" /> : <Plus size={16} />}</button>
          <button className="icon-button mobile-close" aria-label="Close sidebar" onClick={onClose}><X size={16} /></button>
        </>}>
        <div className={`sources-drop ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false) }} onDrop={e => { e.preventDefault(); setDragging(false); if (!busy) onAdd(Array.from(e.dataTransfer.files)) }}>
          {sources.length ? <>
            <label className="select-all"><input type="checkbox" checked={selected.length === sources.length} onChange={onSelectAll} />Use all sources</label>
            <ul className="source-list">{sources.map(source => <li key={source.id}>
              <input type="checkbox" checked={selected.includes(source.id)} onChange={() => onToggle(source.id)} aria-label={`Use ${source.name}`} />
              <FileText size={16} />
              <div className="min-w-0 flex-1"><p title={source.name}>{source.name}</p><small>{(source.size / 1024).toFixed(0)} KB · {source.status === 'staged' ? 'Staged locally' : source.status}</small>{source.error && <small>{source.error}</small>}</div>
              <button className="icon-button delete-button" aria-label={`Delete ${source.name}`} title="Delete source" disabled={busy} onClick={() => onRemove(source.id)}><Trash2 size={14} /></button>
            </li>)}</ul>
            <p className="source-footnote">Ticked sources are used for your next question. Drop files here to add more.</p>
          </> : <div className="source-empty">
            <div className="empty-icon"><Upload size={22} strokeWidth={1.4} /></div>
            <h3>Your material, all together.</h3>
            <p>Drop lecture notes, textbook chapters or revision guides here. PDF, DOCX, TXT or MD, up to 20 MB each.</p>
            <button onClick={() => input.current?.click()} className="text-link" disabled={busy}>Choose your first source</button>
          </div>}
        </div>
      </SidebarSection>

      {bothOpen && <div className="sidebar-resizer" role="separator" aria-orientation="horizontal" aria-label="Resize sources and chats" aria-valuemin={20} aria-valuemax={80} aria-valuenow={Math.round(layout.split * 100)} tabIndex={0} onPointerDown={startResize} onKeyDown={resizeWithKeys} onDoubleClick={() => onLayoutChange(current => ({ ...current, split: 0.5 }))} />}

      <SidebarSection title="Chats" count={chats.length} open={layout.chatsOpen} onToggle={() => onLayoutChange(current => ({ ...current, chatsOpen: !current.chatsOpen }))}
        style={sectionStyle(layout.chatsOpen, layout.sourcesOpen, 1 - layout.split)}
        actions={<button className="icon-button" disabled={!activeChatId} onClick={onNewChat} aria-label="New chat" title="New chat"><SquarePen size={15} /></button>}>
        <ChatHistoryList chats={chats} activeId={activeChatId} onSelect={id => { onSelectChat(id); onClose() }} onDelete={onDeleteChat} />
      </SidebarSection>
    </div>
    <div className="local-notice"><span className="status-dot" />Preview mode · files stay on your device</div>
  </aside>
}
