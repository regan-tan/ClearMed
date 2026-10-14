import { FileText, NotebookPen, X } from 'lucide-react'
import { useCallback, useRef, useState, type CSSProperties } from 'react'
import { notebookApi, type Citation } from './api/notebook'
import type { EmbeddingModel } from './api/types'
import { AppHeader, type Mode } from './components/AppHeader'
import { ColumnResizer } from './components/ColumnResizer'
import { ConfirmDialog } from './components/ConfirmDialog'
import { FlashcardsView } from './components/FlashcardsView'
import { NotebookChat } from './components/NotebookChat'
import { NotebookSidebar, type SidebarLayout } from './components/NotebookSidebar'
import { NotesPanel } from './components/NotesPanel'
import { PracticeView } from './components/PracticeView'
import { useNotebook } from './hooks/useNotebook'
import { createNotebookMeta, DEFAULT_TITLE, deleteNotebookData, notebookStats, type NotebookMeta } from './notebooks'
import { usePersistentState } from './storage'
import { useTheme } from './theme'
import { plural } from './utils'

export default function App() {
  const { theme, toggle } = useTheme()
  const [mode, setMode] = usePersistentState<Mode>('mode', 'notebook')
  const [model, setModel] = usePersistentState<EmbeddingModel>('model', 'e5')
  const [topK, setTopK] = usePersistentState('top-k', 3)
  const [sidebarLayout, setSidebarLayout] = usePersistentState<SidebarLayout>('sidebar-layout', {
    sourcesOpen: true,
    chatsOpen: true,
    split: 0.55,
  })
  const [columns, setColumns] = usePersistentState('column-widths', DEFAULT_COLUMNS)
  const layoutRef = useRef<HTMLElement>(null)

  // Fallback only matters if storage is unavailable; ensureNotebooks() normally seeds the list at startup.
  const [fallbackNotebook] = useState(createNotebookMeta)
  const [storedNotebooks, setNotebooks] = usePersistentState<NotebookMeta[]>('notebooks', [])
  const notebooks = storedNotebooks.length ? storedNotebooks : [fallbackNotebook]
  const [activeNotebookId, setActiveNotebookId] = usePersistentState('active-notebook', '')
  const notebook = notebooks.find(item => item.id === activeNotebookId) ?? notebooks[0]

  const retrieval = { embedding_model: model, top_k: topK }
  const [error, setError] = useState<string | null>(null)
  const nb = useNotebook(notebook.id, retrieval, setError)

  const [citation, setCitation] = useState<Citation | null>(null)
  const [mobilePanel, setMobilePanel] = useState<'sources' | 'notes' | null>(null)
  const [pendingSourceId, setPendingSourceId] = useState<string | null>(null)
  const [confirmingNotebookDelete, setConfirmingNotebookDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const closeSourceDialog = useCallback(() => setPendingSourceId(null), [])
  const closeNotebookDialog = useCallback(() => setConfirmingNotebookDelete(false), [])
  const pendingSource = nb.sources.find(source => source.id === pendingSourceId)

  function switchNotebook(id: string) {
    setActiveNotebookId(id)
    setCitation(null)
    setError(null)
    setMobilePanel(null)
  }

  async function createNotebook() {
    try {
      const created = await notebookApi.createNotebook(DEFAULT_TITLE)
      setNotebooks([...notebooks, { id: created.id, title: created.title, createdAt: created.createdAt }])
      switchNotebook(created.id)
      setMode('notebook')
    } catch {
      setError('Could not create a notebook. Please try again.')
    }
  }

  function renameNotebook(title: string) {
    setNotebooks(notebooks.map(item => (item.id === notebook.id ? { ...item, title } : item)))
    notebookApi.renameNotebook(notebook.id, title).catch(() => setError('Could not save the notebook title.'))
  }

  async function deleteNotebook() {
    setDeleting(true)
    try {
      await notebookApi.deleteNotebook(notebook.id)
      deleteNotebookData(notebook.id)
      const remaining = notebooks.filter(item => item.id !== notebook.id)
      const next = remaining.length ? remaining : [createNotebookMeta()]
      setNotebooks(next)
      switchNotebook(next[0].id)
    } catch {
      setError('Could not delete the notebook. Please try again.')
    } finally {
      setDeleting(false)
      setConfirmingNotebookDelete(false)
    }
  }

  async function deleteSource(id: string, deleteCards: boolean) {
    setDeleting(true)
    await nb.removeSource(id, deleteCards)
    setDeleting(false)
    setPendingSourceId(null)
  }

  return (
    <div className="notebook-app">
      <AppHeader
        notebooks={notebooks}
        notebookId={notebook.id}
        onRenameNotebook={renameNotebook}
        onSelectNotebook={switchNotebook}
        onCreateNotebook={createNotebook}
        onDeleteNotebook={() => setConfirmingNotebookDelete(true)}
        mode={mode}
        onModeChange={setMode}
        model={model}
        topK={topK}
        onModelChange={setModel}
        onTopKChange={setTopK}
        theme={theme}
        onToggleTheme={toggle}
      />

      {error && (
        <div className="upload-error" role="alert">
          {error}
          <button className="icon-button" onClick={() => setError(null)} aria-label="Dismiss error">
            <X size={16} />
          </button>
        </div>
      )}

      {/* Kept mounted so MCQ progress survives switching modes */}
      <div className="mode-view" hidden={mode !== 'practice'}>
        <PracticeView model={model} topK={topK} flashcardOrigins={nb.cardOrigins} onFlashcard={nb.addCard} />
      </div>

      {mode === 'flashcards' && (
        <div className="mode-view">
          <FlashcardsView
            key={notebook.id}
            notebookId={notebook.id}
            cards={nb.cards}
            setCards={nb.setCards}
            sources={nb.sources}
            selected={nb.selected}
            onToggleSource={nb.toggleSource}
            onAddSources={nb.addSources}
            onRemoveSource={setPendingSourceId}
            uploading={nb.uploading}
            retrieval={retrieval}
          />
        </div>
      )}

      {mode === 'notebook' && (
        <>
          <div className="mobile-panel-tabs">
            <button
              onClick={() => setMobilePanel(mobilePanel === 'sources' ? null : 'sources')}
              aria-expanded={mobilePanel === 'sources'}
            >
              <FileText size={15} />
              Sources & chats
            </button>
            <button
              onClick={() => setMobilePanel(mobilePanel === 'notes' ? null : 'notes')}
              aria-expanded={mobilePanel === 'notes'}
            >
              <NotebookPen size={15} />
              Notes ({nb.notes.length})
            </button>
          </div>
          <main
            className="notebook-layout"
            id="chat"
            ref={layoutRef}
            style={{ '--sidebar-w': `${columns.sidebar}px`, '--notes-w': `${columns.notes}px` } as CSSProperties}
          >
            <div className={`sources-slot ${mobilePanel === 'sources' ? 'mobile-visible' : ''}`}>
              <NotebookSidebar
                sources={nb.sources}
                selected={nb.selected}
                busy={nb.uploading}
                onAdd={nb.addSources}
                onToggle={nb.toggleSource}
                onSelectAll={nb.toggleAllSources}
                onRemove={setPendingSourceId}
                onClose={() => setMobilePanel(null)}
                chats={nb.chats}
                activeChatId={nb.activeChatId}
                onSelectChat={nb.selectChat}
                onNewChat={nb.newChat}
                onDeleteChat={nb.deleteChat}
                layout={sidebarLayout}
                onLayoutChange={setSidebarLayout}
              />
            </div>
            <ColumnResizer
              label="Resize sources and chats panel"
              side="left"
              width={columns.sidebar}
              min={MIN_COLUMNS.sidebar}
              getMax={() => columnMax(layoutRef.current, '.sources-slot')}
              defaultWidth={DEFAULT_COLUMNS.sidebar}
              onChange={sidebar => setColumns(current => ({ ...current, sidebar }))}
            />
            <NotebookChat
              chats={nb.chats}
              activeId={nb.activeChatId}
              sourceCount={nb.selected.length}
              busy={nb.sending}
              error={nb.chatError}
              onSend={nb.send}
              onSave={message => nb.addNote(message.content, message.id)}
              onFlashcard={nb.addCardFromMessage}
              onCitation={value => {
                setCitation(value)
                setMobilePanel('notes')
              }}
              onSelectChat={nb.selectChat}
              onNewChat={nb.newChat}
              onDeleteChat={nb.deleteChat}
              onAdd={() => {
                setMobilePanel('sources')
                document.querySelector<HTMLInputElement>('input[type="file"]')?.click()
              }}
              savedIds={nb.notes.map(note => note.id)}
              flashcardIds={nb.cardOrigins.flatMap(origin => (origin.startsWith('chat:') ? [origin.slice(5)] : []))}
            />
            <ColumnResizer
              label="Resize notes panel"
              side="right"
              width={columns.notes}
              min={MIN_COLUMNS.notes}
              getMax={() => columnMax(layoutRef.current, '.notes-panel')}
              defaultWidth={DEFAULT_COLUMNS.notes}
              onChange={notes => setColumns(current => ({ ...current, notes }))}
            />
            <NotesPanel
              key={notebook.id}
              notes={nb.notes}
              citation={citation}
              mobileVisible={mobilePanel === 'notes'}
              onAddNote={nb.addNote}
              onRemoveNote={nb.removeNote}
              onCloseCitation={() => setCitation(null)}
              onClose={() => setMobilePanel(null)}
            />
          </main>
        </>
      )}

      {pendingSource && (
        <ConfirmDialog
          title="Delete this source?"
          confirmLabel="Delete source"
          busy={deleting}
          option={sourceCardsOption(nb.cards.filter(card => card.sourceId === pendingSource.id).length)}
          onConfirm={deleteCards => deleteSource(pendingSource.id, deleteCards)}
          onCancel={closeSourceDialog}
        >
          <strong>{pendingSource.name}</strong> will be removed from this notebook and won’t be used to answer
          questions.
        </ConfirmDialog>
      )}
      {confirmingNotebookDelete && (
        <ConfirmDialog
          title="Delete this notebook?"
          confirmLabel="Delete notebook"
          busy={deleting}
          onConfirm={deleteNotebook}
          onCancel={closeNotebookDialog}
        >
          <strong>{notebook.title}</strong> and everything in it will be deleted:{' '}
          {describeContents(notebook.id, nb.notes.length)}. This can’t be undone.
        </ConfirmDialog>
      )}
    </div>
  )
}

const DEFAULT_COLUMNS = { sidebar: 280, notes: 280 }
const MIN_COLUMNS = { sidebar: 200, notes: 220 }
const MAX_COLUMN = 640
const MIN_CHAT = 360

// A side panel may only grow into the space the chat column can spare.
function columnMax(layout: HTMLElement | null, panelSelector: string) {
  const panel = layout?.querySelector<HTMLElement>(panelSelector)
  const chat = layout?.querySelector<HTMLElement>('.notebook-chat')
  if (!panel || !chat) return MAX_COLUMN
  return Math.min(MAX_COLUMN, panel.offsetWidth + chat.offsetWidth - MIN_CHAT)
}

function sourceCardsOption(cardCount: number) {
  return cardCount ? `Also delete the ${plural(cardCount, 'flashcard')} made from it` : undefined
}

function describeContents(notebookId: string, noteCount: number) {
  const stats = notebookStats(notebookId)
  return `${plural(stats.sources, 'source')}, ${plural(stats.chats, 'chat')}, ${plural(noteCount, 'note')} and ${plural(stats.cards, 'flashcard')}`
}
