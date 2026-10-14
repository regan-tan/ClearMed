import { BookOpen, FileText, Moon, NotebookPen, Plus, SlidersHorizontal, Sun, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { notebookApi, type Citation, type Conversation, type Note, type NotebookMessage, type NotebookSource } from './api/notebook'
import type { EmbeddingModel } from './api/types'
import { ConfirmDialog } from './components/ConfirmDialog'
import { FlashcardsView } from './components/FlashcardsView'
import { NotebookChat } from './components/NotebookChat'
import { NotebookSidebar, type SidebarLayout } from './components/NotebookSidebar'
import { NotebookSwitcher } from './components/NotebookSwitcher'
import { PracticeView } from './components/PracticeView'
import { StudioPanel } from './components/StudioPanel'
import { createCard, type Flashcard, type FlashcardDraft } from './flashcards'
import { createNotebookMeta, DEFAULT_TITLE, deleteNotebookData, notebookKey, notebookStats, type NotebookMeta } from './notebooks'
import { usePersistentState } from './storage'
import { useTheme } from './theme'

type Mode = 'notebook' | 'practice' | 'flashcards'

function chatTitle(message: string) {
  const line = message.split('\n')[0].trim()
  return line.length > 60 ? `${line.slice(0, 57)}…` : line
}

export default function App() {
  const { theme, toggle } = useTheme()
  const [mode, setMode] = usePersistentState<Mode>('mode', 'notebook')
  const [model, setModel] = usePersistentState<EmbeddingModel>('model', 'e5')
  const [topK, setTopK] = usePersistentState('top-k', 3)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [sidebarLayout, setSidebarLayout] = usePersistentState<SidebarLayout>('sidebar-layout', { sourcesOpen: true, chatsOpen: true, split: 0.55 })
  const settingsRef = useRef<HTMLDivElement>(null)
  const retrieval = { embedding_model: model, top_k: topK }
  // Fallback only matters if storage is unavailable; ensureNotebooks() normally seeds the list at startup.
  const [fallbackNotebook] = useState(createNotebookMeta)
  const [storedNotebooks, setNotebooks] = usePersistentState<NotebookMeta[]>('notebooks', [])
  const notebooks = storedNotebooks.length ? storedNotebooks : [fallbackNotebook]
  const [activeNotebookId, setActiveNotebookId] = usePersistentState('active-notebook', '')
  const notebook = notebooks.find(item => item.id === activeNotebookId) ?? notebooks[0]
  const nb = notebook.id
  // Per-notebook data. Source metadata only: the preview never reads file contents.
  const [sources, setSources] = usePersistentState<NotebookSource[]>(notebookKey(nb, 'sources'), [])
  const [selected, setSelected] = usePersistentState<string[]>(notebookKey(nb, 'selected-sources'), [])
  const [chats, setChats] = usePersistentState<Conversation[]>(notebookKey(nb, 'chats'), [])
  const [activeChatId, setActiveChatId] = usePersistentState(notebookKey(nb, 'active-chat'), '')
  const [notes, setNotes] = usePersistentState<Note[]>(notebookKey(nb, 'notes'), [])
  const [cards, setCards] = usePersistentState<Flashcard[]>(notebookKey(nb, 'flashcards'), [])
  const [citation, setCitation] = useState<Citation | null>(null)
  const [mobilePanel, setMobilePanel] = useState<'sources' | 'notes' | null>(null)
  const [uploading, setUploading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [deletingNotebook, setDeletingNotebook] = useState(false)
  const [removing, setRemoving] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  const [noteDraft, setNoteDraft] = useState('')
  const [editingNote, setEditingNote] = useState(false)
  const requestLock = useRef(false)
  const uploadLock = useRef(false)

  const pendingSource = sources.find(source => source.id === pendingDelete)
  const activeChat = chats.find(chat => chat.id === activeChatId)
  // Close the settings popover on Escape or a click outside it.
  useEffect(() => {
    if (!settingsOpen) return
    function onPointer(e: MouseEvent) { if (!settingsRef.current?.contains(e.target as Node)) setSettingsOpen(false) }
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') setSettingsOpen(false) }
    document.addEventListener('mousedown', onPointer); document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onPointer); document.removeEventListener('keydown', onKey) }
  }, [settingsOpen])
  const closeDeleteDialog = useCallback(() => setPendingDelete(null), [])
  const closeNotebookDialog = useCallback(() => setDeletingNotebook(false), [])
  const cardOrigins = cards.flatMap(card => card.origin ? [card.origin] : [])

  function switchNotebook(id: string) {
    setActiveNotebookId(id)
    setCitation(null); setChatError(null); setError(null); setMobilePanel(null); setEditingNote(false); setNoteDraft('')
  }
  async function createNotebook() {
    try {
      const created = await notebookApi.createNotebook(DEFAULT_TITLE)
      setNotebooks([...notebooks, { id: created.id, title: created.title, createdAt: created.createdAt }])
      switchNotebook(created.id)
      setMode('notebook')
    } catch { setError('Could not create a notebook. Please try again.') }
  }
  function renameNotebook(title: string) {
    setNotebooks(notebooks.map(item => item.id === nb ? { ...item, title } : item))
    notebookApi.renameNotebook(nb, title).catch(() => setError('Could not save the notebook title.'))
  }
  async function deleteNotebook() {
    setRemoving(true)
    try {
      await notebookApi.deleteNotebook(nb)
      deleteNotebookData(nb)
      const remaining = notebooks.filter(item => item.id !== nb)
      const next = remaining.length ? remaining : [createNotebookMeta()]
      setNotebooks(next)
      switchNotebook(next[0].id)
    } catch { setError('Could not delete the notebook. Please try again.') }
    finally { setRemoving(false); setDeletingNotebook(false) }
  }
  function addCard(draft: FlashcardDraft) {
    setCards(current => draft.origin && current.some(card => card.origin === draft.origin) ? current : [createCard(draft), ...current])
  }
  async function addSources(files: File[]) {
    if (uploadLock.current || !files.length) return
    uploadLock.current = true; setUploading(true); setError(null)
    const failures: string[] = []
    try {
      for (const file of files) {
        if (!/\.(pdf|docx|txt|md)$/i.test(file.name)) { failures.push(`${file.name}: unsupported file type.`); continue }
        if (!file.size || file.size > 20 * 1024 * 1024) { failures.push(`${file.name}: choose a non-empty file under 20 MB.`); continue }
        try {
          const source = await notebookApi.uploadSource(nb, file)
          setSources(current => [...current, source])
          setSelected(current => [...current, source.id])
        } catch { failures.push(`${file.name}: couldn't add this source. Please try again.`) }
      }
      if (failures.length) setError(failures.join(' '))
    } finally { uploadLock.current = false; setUploading(false) }
  }
  async function removeSource(id: string, deleteCards: boolean) {
    setRemoving(true)
    try {
      await notebookApi.removeSource(nb, id)
      setSources(current => current.filter(source => source.id !== id))
      setSelected(current => current.filter(sourceId => sourceId !== id))
      // Kept cards stay revisable but no longer ground follow-up questions in the deleted source.
      setCards(current => deleteCards ? current.filter(card => card.sourceId !== id) : current.map(card => card.sourceId === id ? { ...card, sourceId: undefined } : card))
      setPendingDelete(null)
    } catch {
      setPendingDelete(null)
      setError('Could not remove the source. Please try again.')
    } finally { setRemoving(false) }
  }
  function toggleSource(id: string) {
    setSelected(current => current.includes(id) ? current.filter(item => item !== id) : [...current, id])
  }
  function updateChat(id: string, update: (messages: NotebookMessage[]) => NotebookMessage[]) {
    setChats(current => current.map(chat => chat.id === id ? { ...chat, messages: update(chat.messages), updatedAt: Date.now() } : chat))
  }
  async function send(message: string) {
    if (requestLock.current) return
    requestLock.current = true; setSending(true); setChatError(null)
    const user: NotebookMessage = { id: crypto.randomUUID(), role: 'user', content: message }
    const history = (activeChat?.messages ?? []).map(({ role, content }) => ({ role, content }))
    // Capture the target chat so the reply lands there even if the user switches chats meanwhile.
    const chatId = activeChat?.id ?? crypto.randomUUID()
    if (activeChat) updateChat(chatId, messages => [...messages, user])
    else { setChats(current => [...current, { id: chatId, title: chatTitle(message), messages: [user], updatedAt: Date.now() }]); setActiveChatId(chatId) }
    try {
      const response = await notebookApi.chat(nb, { message, source_ids: [...selected], history, retrieval })
      updateChat(chatId, messages => [...messages, { id: crypto.randomUUID(), role: 'assistant', ...response }])
    } catch {
      updateChat(chatId, messages => messages.filter(item => item.id !== user.id))
      setChats(current => current.filter(chat => chat.messages.length))
      setChatError(`Couldn’t send your question. Please try again: “${message}”`)
    } finally { requestLock.current = false; setSending(false) }
  }
  function selectChat(id: string) { setActiveChatId(id); setChatError(null) }
  function newChat() { setActiveChatId(''); setChatError(null) }
  function deleteChat(id: string) {
    setChats(current => current.filter(chat => chat.id !== id))
    if (id === activeChatId) setActiveChatId('')
  }
  function flashcardFromMessage(message: NotebookMessage) {
    const messages = activeChat?.messages ?? []
    const question = messages.slice(0, messages.findIndex(item => item.id === message.id)).findLast(item => item.role === 'user')
    addCard({ front: question?.content ?? activeChat?.title ?? 'Explain this concept', back: message.content, label: 'From chat', origin: `chat:${message.id}` })
  }

  const modes: { value: Mode; label: ReactNode }[] = [
    { value: 'notebook', label: 'Notebook' },
    { value: 'practice', label: <>Practice<span className="mode-full"> MCQ</span></> },
    { value: 'flashcards', label: 'Flashcards' },
  ]

  return <div className="notebook-app">
    <header className="notebook-header">
      <div className="header-start"><a href="#chat" className="notebook-brand"><span className="brand-symbol"><BookOpen size={22} /></span>ClearMed</a><span className="header-divider" /><NotebookSwitcher notebooks={notebooks} activeId={nb} onRename={renameNotebook} onSelect={switchNotebook} onCreate={createNotebook} onDelete={() => setDeletingNotebook(true)} /></div>
      <div className="mode-switch" role="group" aria-label="Mode">{modes.map(({ value, label }) => <button key={value} aria-pressed={mode === value} onClick={() => { setMode(value); setSettingsOpen(false) }}>{label}</button>)}</div>
      <div className="header-actions"><span className="preview-label">Frontend preview</span><div className="settings-anchor" ref={settingsRef}><button className="icon-button" onClick={() => setSettingsOpen(open => !open)} aria-label="Search settings" aria-expanded={settingsOpen} title="Search settings"><SlidersHorizontal size={18} /></button>{settingsOpen && <div className="settings-popover"><StudioPanel model={model} topK={topK} onModelChange={setModel} onTopKChange={setTopK} onCollapse={() => setSettingsOpen(false)} /></div>}</div><button className="icon-button" onClick={toggle} aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>{theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}</button></div>
    </header>
    {error && <div className="upload-error" role="alert">{error}<button className="icon-button" onClick={() => setError(null)} aria-label="Dismiss error"><X size={16} /></button></div>}
    {/* Kept mounted so MCQ progress survives switching modes */}
    <div className="mode-view" hidden={mode !== 'practice'}><PracticeView model={model} topK={topK} flashcardOrigins={cardOrigins} onFlashcard={addCard} /></div>
    {mode === 'flashcards' && <div className="mode-view"><FlashcardsView key={nb} notebookId={nb} cards={cards} setCards={setCards} sources={sources} selected={selected} onToggleSource={toggleSource} onAddSources={addSources} onRemoveSource={setPendingDelete} uploading={uploading} retrieval={retrieval} /></div>}
    {mode === 'notebook' && <>
    <div className="mobile-panel-tabs"><button onClick={() => setMobilePanel(mobilePanel === 'sources' ? null : 'sources')} aria-expanded={mobilePanel === 'sources'}><FileText size={15} />Sources & chats</button><button onClick={() => setMobilePanel(mobilePanel === 'notes' ? null : 'notes')} aria-expanded={mobilePanel === 'notes'}><NotebookPen size={15} />Notes ({notes.length})</button></div>
    <main className="notebook-layout" id="chat">
      <div className={`sources-slot ${mobilePanel === 'sources' ? 'mobile-visible' : ''}`}><NotebookSidebar sources={sources} selected={selected} busy={uploading} onAdd={addSources} onToggle={toggleSource} onSelectAll={() => setSelected(selected.length === sources.length ? [] : sources.map(source => source.id))} onRemove={setPendingDelete} onClose={() => setMobilePanel(null)}
        chats={chats} activeChatId={activeChatId} onSelectChat={selectChat} onNewChat={newChat} onDeleteChat={deleteChat} layout={sidebarLayout} onLayoutChange={setSidebarLayout} /></div>
      <NotebookChat chats={chats} activeId={activeChatId} sourceCount={selected.length} busy={sending} error={chatError} onSend={send}
        onSave={message => setNotes(current => current.some(note => note.id === message.id) ? current : [...current, { id: message.id, text: message.content }])}
        onFlashcard={flashcardFromMessage} onCitation={value => { setCitation(value); setMobilePanel('notes') }}
        onSelectChat={selectChat} onNewChat={newChat} onDeleteChat={deleteChat}
        onAdd={() => { setMobilePanel('sources'); document.querySelector<HTMLInputElement>('input[type="file"]')?.click() }}
        savedIds={notes.map(note => note.id)} flashcardIds={cardOrigins.flatMap(origin => origin.startsWith('chat:') ? [origin.slice(5)] : [])} />
      <aside className={`notebook-panel notes-panel ${mobilePanel === 'notes' ? 'mobile-visible' : ''}`}><div className="panel-heading"><h2>{citation ? 'Source passage' : <>Notes <span className="count-badge">{notes.length}</span></>}</h2><button className={`icon-button ${citation ? '' : 'mobile-close'}`} aria-label={citation ? 'Close source passage' : 'Close notes'} onClick={() => { setCitation(null); setMobilePanel(null) }}><X size={16} /></button></div>
        {citation ? <div className="citation-preview"><FileText size={22} /><h3>{citation.source_name}</h3>{citation.page && <p>Page {citation.page}</p>}<blockquote>{citation.excerpt}</blockquote><button className="text-link" onClick={() => setCitation(null)}>Back to notes</button></div> : <><div className="notes-intro"><p>Keep what clicks.</p><span>Save useful explanations or add your own study notes.</span></div><button className="add-note" onClick={() => setEditingNote(true)}><Plus size={16} />Write a note</button>{editingNote && <form className="note-editor" onSubmit={e => { e.preventDefault(); if (!noteDraft.trim()) return; setNotes(current => [...current, { id: crypto.randomUUID(), text: noteDraft.trim() }]); setNoteDraft(''); setEditingNote(false) }}><textarea autoFocus aria-label="New study note" placeholder="What do you want to remember?" value={noteDraft} onChange={e => setNoteDraft(e.target.value)} /><div><button type="button" onClick={() => { setEditingNote(false); setNoteDraft('') }}>Cancel</button><button type="submit" disabled={!noteDraft.trim()}>Save note</button></div></form>}
          <div className="notes-list">{notes.length ? notes.map(note => <article className="study-note" key={note.id}><div><NotebookPen size={15} /><button className="icon-button" aria-label="Remove note" onClick={() => setNotes(current => current.filter(item => item.id !== note.id))}><X size={13} /></button></div><p>{note.text}</p></article>) : !editingNote && <div className="notes-empty"><NotebookPen size={26} strokeWidth={1.2} /><p>A little space for<br />the important bits.</p></div>}</div><p className="notes-footer">Saved in this browser. Clearing site data removes it.</p></>}
      </aside>
    </main>
    </>}
    {pendingSource && (() => {
      const cardCount = cards.filter(card => card.sourceId === pendingSource.id).length
      return <ConfirmDialog title="Delete this source?" confirmLabel="Delete source" busy={removing} option={cardCount ? `Also delete the ${cardCount} flashcard${cardCount !== 1 ? 's' : ''} made from it` : undefined} onConfirm={deleteCards => removeSource(pendingSource.id, deleteCards)} onCancel={closeDeleteDialog}>
        <strong>{pendingSource.name}</strong> will be removed from this notebook and won’t be used to answer questions.
      </ConfirmDialog>
    })()}
    {deletingNotebook && (() => {
      const stats = notebookStats(nb)
      return <ConfirmDialog title="Delete this notebook?" confirmLabel="Delete notebook" busy={removing} onConfirm={deleteNotebook} onCancel={closeNotebookDialog}>
        <strong>{notebook.title}</strong> and everything in it will be deleted: {stats.sources} source{stats.sources !== 1 ? 's' : ''}, {stats.chats} chat{stats.chats !== 1 ? 's' : ''}, {notes.length} note{notes.length !== 1 ? 's' : ''} and {stats.cards} flashcard{stats.cards !== 1 ? 's' : ''}. This can’t be undone.
      </ConfirmDialog>
    })()}
  </div>
}
