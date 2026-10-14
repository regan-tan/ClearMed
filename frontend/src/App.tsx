import { BookOpen, FileText, Moon, NotebookPen, Plus, SlidersHorizontal, Sun, X } from 'lucide-react'
import { useCallback, useRef, useState, type ReactNode } from 'react'
import { notebookApi, type Citation, type Conversation, type Note, type NotebookMessage, type NotebookSource } from './api/notebook'
import type { EmbeddingModel } from './api/types'
import { DeleteSourceDialog } from './components/DeleteSourceDialog'
import { FlashcardsView } from './components/FlashcardsView'
import { NotebookChat } from './components/NotebookChat'
import { PracticeView } from './components/PracticeView'
import { SourcesPanel } from './components/SourcesPanel'
import { StudioPanel } from './components/StudioPanel'
import { createCard, type Flashcard, type FlashcardDraft } from './flashcards'
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
  const [title, setTitle] = usePersistentState('title', 'Untitled notebook')
  // Source metadata only: the preview never reads file contents. Use notebookApi.listSources() once a backend exists.
  const [sources, setSources] = usePersistentState<NotebookSource[]>('sources', [])
  const [selected, setSelected] = usePersistentState<string[]>('selected-sources', [])
  const [chats, setChats] = usePersistentState<Conversation[]>('chats', [])
  const [activeChatId, setActiveChatId] = usePersistentState('active-chat', '')
  const [notes, setNotes] = usePersistentState<Note[]>('notes', [])
  const [cards, setCards] = usePersistentState<Flashcard[]>('flashcards', [])
  const [citation, setCitation] = useState<Citation | null>(null)
  const [mobilePanel, setMobilePanel] = useState<'sources' | 'notes' | null>(null)
  const [uploading, setUploading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [removing, setRemoving] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  const [noteDraft, setNoteDraft] = useState('')
  const [editingNote, setEditingNote] = useState(false)
  const requestLock = useRef(false)
  const uploadLock = useRef(false)

  const pendingSource = sources.find(source => source.id === pendingDelete)
  const activeChat = chats.find(chat => chat.id === activeChatId)
  const closeDeleteDialog = useCallback(() => setPendingDelete(null), [])
  const cardOrigins = cards.flatMap(card => card.origin ? [card.origin] : [])

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
          const source = await notebookApi.uploadSource(file)
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
      await notebookApi.removeSource(id)
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
      const response = await notebookApi.chat({ message, source_ids: [...selected], history })
      updateChat(chatId, messages => [...messages, { id: crypto.randomUUID(), role: 'assistant', ...response }])
    } catch {
      updateChat(chatId, messages => messages.filter(item => item.id !== user.id))
      setChats(current => current.filter(chat => chat.messages.length))
      setChatError(`Couldn’t send your question. Please try again: “${message}”`)
    } finally { requestLock.current = false; setSending(false) }
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
      <div className="header-start"><a href="#chat" className="notebook-brand"><span className="brand-symbol"><BookOpen size={22} /></span><span className="brand-name">ClearMed</span></a><span className="header-divider" /><input className="notebook-title" aria-label="Notebook title" value={title} onChange={e => setTitle(e.target.value)} onBlur={() => { if (!title.trim()) setTitle('Untitled notebook') }} /></div>
      <div className="mode-switch" role="group" aria-label="Mode">{modes.map(({ value, label }) => <button key={value} aria-pressed={mode === value} onClick={() => { setMode(value); setSettingsOpen(false) }}>{label}</button>)}</div>
      <div className="header-actions"><span className="preview-label">Frontend preview</span>{mode === 'practice' && <div className="settings-anchor"><button className="icon-button" onClick={() => setSettingsOpen(open => !open)} aria-label="Evidence settings" aria-expanded={settingsOpen} title="Evidence settings"><SlidersHorizontal size={18} /></button>{settingsOpen && <div className="settings-popover"><StudioPanel model={model} topK={topK} onModelChange={setModel} onTopKChange={setTopK} onCollapse={() => setSettingsOpen(false)} /></div>}</div>}<button className="icon-button" onClick={toggle} aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>{theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}</button></div>
    </header>
    {error && <div className="upload-error" role="alert">{error}<button className="icon-button" onClick={() => setError(null)} aria-label="Dismiss error"><X size={16} /></button></div>}
    {/* Kept mounted so MCQ progress survives switching modes */}
    <div className="mode-view" hidden={mode !== 'practice'}><PracticeView model={model} topK={topK} flashcardOrigins={cardOrigins} onFlashcard={addCard} /></div>
    {mode === 'flashcards' && <div className="mode-view"><FlashcardsView cards={cards} setCards={setCards} sources={sources} selected={selected} onToggleSource={toggleSource} onAddSources={addSources} onRemoveSource={setPendingDelete} uploading={uploading} /></div>}
    {mode === 'notebook' && <>
    <div className="mobile-panel-tabs"><button onClick={() => setMobilePanel(mobilePanel === 'sources' ? null : 'sources')} aria-expanded={mobilePanel === 'sources'}><FileText size={15} />Sources ({sources.length})</button><button onClick={() => setMobilePanel(mobilePanel === 'notes' ? null : 'notes')} aria-expanded={mobilePanel === 'notes'}><NotebookPen size={15} />Notes ({notes.length})</button></div>
    <main className="notebook-layout" id="chat">
      <div className={`sources-slot ${mobilePanel === 'sources' ? 'mobile-visible' : ''}`}><SourcesPanel sources={sources} selected={selected} busy={uploading} onAdd={addSources} onToggle={toggleSource} onSelectAll={() => setSelected(selected.length === sources.length ? [] : sources.map(source => source.id))} onRemove={setPendingDelete} onClose={() => setMobilePanel(null)} /></div>
      <NotebookChat chats={chats} activeId={activeChatId} sourceCount={selected.length} busy={sending} error={chatError} onSend={send}
        onSave={message => setNotes(current => current.some(note => note.id === message.id) ? current : [...current, { id: message.id, text: message.content }])}
        onFlashcard={flashcardFromMessage} onCitation={value => { setCitation(value); setMobilePanel('notes') }}
        onSelectChat={id => { setActiveChatId(id); setChatError(null) }} onNewChat={() => { setActiveChatId(''); setChatError(null) }}
        onDeleteChat={id => { setChats(current => current.filter(chat => chat.id !== id)); if (id === activeChatId) setActiveChatId('') }}
        onAdd={() => { setMobilePanel('sources'); document.querySelector<HTMLInputElement>('input[type="file"]')?.click() }}
        savedIds={notes.map(note => note.id)} flashcardIds={cardOrigins.flatMap(origin => origin.startsWith('chat:') ? [origin.slice(5)] : [])} />
      <aside className={`notebook-panel notes-panel ${mobilePanel === 'notes' ? 'mobile-visible' : ''}`}><div className="panel-heading"><h2>{citation ? 'Source passage' : <>Notes <span className="count-badge">{notes.length}</span></>}</h2><button className={`icon-button ${citation ? '' : 'mobile-close'}`} aria-label={citation ? 'Close source passage' : 'Close notes'} onClick={() => { setCitation(null); setMobilePanel(null) }}><X size={16} /></button></div>
        {citation ? <div className="citation-preview"><FileText size={22} /><h3>{citation.source_name}</h3>{citation.page && <p>Page {citation.page}</p>}<blockquote>{citation.excerpt}</blockquote><button className="text-link" onClick={() => setCitation(null)}>Back to notes</button></div> : <><div className="notes-intro"><p>Keep what clicks.</p><span>Save useful explanations or add your own study notes.</span></div><button className="add-note" onClick={() => setEditingNote(true)}><Plus size={16} />Write a note</button>{editingNote && <form className="note-editor" onSubmit={e => { e.preventDefault(); if (!noteDraft.trim()) return; setNotes(current => [...current, { id: crypto.randomUUID(), text: noteDraft.trim() }]); setNoteDraft(''); setEditingNote(false) }}><textarea autoFocus aria-label="New study note" placeholder="What do you want to remember?" value={noteDraft} onChange={e => setNoteDraft(e.target.value)} /><div><button type="button" onClick={() => { setEditingNote(false); setNoteDraft('') }}>Cancel</button><button type="submit" disabled={!noteDraft.trim()}>Save note</button></div></form>}
          <div className="notes-list">{notes.length ? notes.map(note => <article className="study-note" key={note.id}><div><NotebookPen size={15} /><button className="icon-button" aria-label="Remove note" onClick={() => setNotes(current => current.filter(item => item.id !== note.id))}><X size={13} /></button></div><p>{note.text}</p></article>) : !editingNote && <div className="notes-empty"><NotebookPen size={26} strokeWidth={1.2} /><p>A little space for<br />the important bits.</p></div>}</div><p className="notes-footer">Saved in this browser. Clearing site data removes it.</p></>}
      </aside>
    </main>
    </>}
    {pendingSource && <DeleteSourceDialog source={pendingSource} cardCount={cards.filter(card => card.sourceId === pendingSource.id).length} busy={removing} onConfirm={deleteCards => removeSource(pendingSource.id, deleteCards)} onCancel={closeDeleteDialog} />}
  </div>
}
