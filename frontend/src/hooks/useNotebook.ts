import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import {
  notebookApi,
  type Conversation,
  type Note,
  type NotebookMessage,
  type NotebookSource,
  type RetrievalSettings,
} from '../api/notebook'
import { createCard, type Flashcard, type FlashcardDraft } from '../flashcards'
import { notebookKey } from '../notebooks'
import { isUsableSource, sourceFileProblem } from '../sources'
import { usePersistentState } from '../storage'

const SOURCE_POLL_MS = 3000

function chatTitle(message: string) {
  const line = message.split('\n')[0].trim()
  return line.length > 60 ? `${line.slice(0, 57)}…` : line
}

/**
 * One notebook's data (sources, chats, notes, flashcards), persisted per notebook, and the actions on it.
 * Async actions capture the notebook id when they start, so results land in that notebook even if the
 * user switches notebooks before they finish.
 */
export function useNotebook(notebookId: string, retrieval: RetrievalSettings, onError: (message: string) => void) {
  const [sources, setSources] = usePersistentState<NotebookSource[]>(notebookKey(notebookId, 'sources'), [])
  const [selected, setSelected] = usePersistentState<string[]>(notebookKey(notebookId, 'selected-sources'), [])
  const [chats, setChats] = usePersistentState<Conversation[]>(notebookKey(notebookId, 'chats'), [])
  const [activeChatId, setActiveChatId] = usePersistentState(notebookKey(notebookId, 'active-chat'), '')
  const [notes, setNotes] = usePersistentState<Note[]>(notebookKey(notebookId, 'notes'), [])
  const [cards, setCards] = usePersistentState<Flashcard[]>(notebookKey(notebookId, 'flashcards'), [])
  const [uploading, setUploading] = useState(false)
  const [sending, setSending] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  const requestLock = useRef(false)
  const uploadLock = useRef(false)

  // Clear the previous notebook's chat error when switching notebooks.
  const [errorNotebook, setErrorNotebook] = useState(notebookId)
  if (errorNotebook !== notebookId) {
    setErrorNotebook(notebookId)
    setChatError(null)
  }

  const activeChat = chats.find(chat => chat.id === activeChatId)
  // Selection may include sources still processing or failed; only usable ones are sent with requests.
  const usableIds = sources.filter(isUsableSource).map(source => source.id)
  const readySelected = selected.filter(id => usableIds.includes(id))

  // With a real backend the server owns the source list: load it whenever a notebook opens.
  // The preview adapter has nothing to load, so locally staged sources are kept as they are.
  // Loading is derived: it lasts until the request for the current notebook and attempt has settled.
  const [sourcesAttempt, setSourcesAttempt] = useState(0)
  const sourcesRequest = `${notebookId}:${sourcesAttempt}`
  const [sourcesResult, setSourcesResult] = useState({ request: '', failed: false })
  const sourcesSettled = notebookApi.preview || sourcesResult.request === sourcesRequest
  useEffect(() => {
    if (notebookApi.preview) return
    let cancelled = false
    notebookApi
      .listSources(notebookId)
      .then(list => {
        if (cancelled) return
        setSources(list)
        setSelected(current => current.filter(id => list.some(source => source.id === id)))
        setSourcesResult({ request: sourcesRequest, failed: false })
      })
      .catch(() => {
        if (!cancelled) setSourcesResult({ request: sourcesRequest, failed: true })
      })
    return () => {
      cancelled = true
    }
  }, [notebookId, sourcesRequest, setSources, setSelected])

  // While any source is processing, poll for status changes and select sources as they become ready.
  const sourcesRef = useRef(sources)
  useLayoutEffect(() => {
    sourcesRef.current = sources
  })
  const processing = sources.some(source => source.status === 'processing')
  useEffect(() => {
    if (!processing) return
    const timer = setInterval(async () => {
      try {
        const latest = await notebookApi.listSources(notebookId)
        const nowReady = sourcesRef.current
          .filter(source => source.status === 'processing')
          .filter(source => latest.some(item => item.id === source.id && item.status === 'ready'))
          .map(source => source.id)
        setSources(current => current.map(source => latest.find(item => item.id === source.id) ?? source))
        if (nowReady.length) setSelected(current => [...current, ...nowReady.filter(id => !current.includes(id))])
      } catch {
        // keep polling; a transient failure shouldn't surface while files are still processing
      }
    }, SOURCE_POLL_MS)
    return () => clearInterval(timer)
  }, [processing, notebookId, setSources, setSelected])
  const cardOrigins = cards.flatMap(card => (card.origin ? [card.origin] : []))

  async function addSources(files: File[]) {
    if (uploadLock.current || !files.length) return
    uploadLock.current = true
    setUploading(true)
    const failures: string[] = []
    try {
      for (const file of files) {
        const problem = sourceFileProblem(file)
        if (problem) {
          failures.push(problem)
          continue
        }
        try {
          const source = await notebookApi.uploadSource(notebookId, file)
          setSources(current => [...current, source])
          if (isUsableSource(source)) setSelected(current => [...current, source.id])
          else if (source.status === 'error') failures.push(`${file.name}: ${source.error ?? 'couldn’t be processed.'}`)
        } catch {
          failures.push(`${file.name}: couldn't add this source. Please try again.`)
        }
      }
      if (failures.length) onError(failures.join(' '))
    } finally {
      uploadLock.current = false
      setUploading(false)
    }
  }

  async function removeSource(id: string, deleteCards: boolean) {
    try {
      await notebookApi.removeSource(notebookId, id)
      setSources(current => current.filter(source => source.id !== id))
      setSelected(current => current.filter(sourceId => sourceId !== id))
      // Kept cards stay revisable but no longer ground follow-up questions in the deleted source.
      setCards(current =>
        deleteCards
          ? current.filter(card => card.sourceId !== id)
          : current.map(card => (card.sourceId === id ? { ...card, sourceId: undefined } : card)),
      )
    } catch {
      onError('Could not remove the source. Please try again.')
    }
  }

  function toggleSource(id: string) {
    setSelected(current => (current.includes(id) ? current.filter(item => item !== id) : [...current, id]))
  }

  function toggleAllSources() {
    setSelected(readySelected.length === usableIds.length ? [] : usableIds)
  }

  function updateChat(id: string, update: (messages: NotebookMessage[]) => NotebookMessage[]) {
    setChats(current =>
      current.map(chat =>
        chat.id === id ? { ...chat, messages: update(chat.messages), updatedAt: Date.now() } : chat,
      ),
    )
  }

  async function send(message: string) {
    if (requestLock.current) return
    requestLock.current = true
    setSending(true)
    setChatError(null)
    const user: NotebookMessage = { id: crypto.randomUUID(), role: 'user', content: message }
    const history = (activeChat?.messages ?? []).map(({ role, content }) => ({ role, content }))
    // Capture the target chat so the reply lands there even if the user switches chats meanwhile.
    const chatId = activeChat?.id ?? crypto.randomUUID()
    if (activeChat) updateChat(chatId, messages => [...messages, user])
    else {
      setChats(current => [
        ...current,
        { id: chatId, title: chatTitle(message), messages: [user], updatedAt: Date.now() },
      ])
      setActiveChatId(chatId)
    }
    try {
      const response = await notebookApi.chat(notebookId, { message, source_ids: readySelected, history, retrieval })
      updateChat(chatId, messages => [...messages, { id: crypto.randomUUID(), role: 'assistant', ...response }])
    } catch {
      updateChat(chatId, messages => messages.filter(item => item.id !== user.id))
      setChats(current => current.filter(chat => chat.messages.length))
      setChatError(`Couldn’t send your question. Please try again: “${message}”`)
    } finally {
      requestLock.current = false
      setSending(false)
    }
  }

  function selectChat(id: string) {
    setActiveChatId(id)
    setChatError(null)
  }

  function newChat() {
    setActiveChatId('')
    setChatError(null)
  }

  function deleteChat(id: string) {
    setChats(current => current.filter(chat => chat.id !== id))
    if (id === activeChatId) setActiveChatId('')
  }

  function addNote(text: string, id: string = crypto.randomUUID()) {
    setNotes(current => (current.some(note => note.id === id) ? current : [...current, { id, text }]))
  }

  function removeNote(id: string) {
    setNotes(current => current.filter(note => note.id !== id))
  }

  function addCard(draft: FlashcardDraft) {
    setCards(current =>
      draft.origin && current.some(card => card.origin === draft.origin) ? current : [createCard(draft), ...current],
    )
  }

  // Card from a chat answer: the question that prompted it on the front, the answer on the back.
  function addCardFromMessage(message: NotebookMessage) {
    const messages = activeChat?.messages ?? []
    const question = messages
      .slice(
        0,
        messages.findIndex(item => item.id === message.id),
      )
      .findLast(item => item.role === 'user')
    addCard({
      front: question?.content ?? activeChat?.title ?? 'Explain this concept',
      back: message.content,
      label: 'From chat',
      origin: `chat:${message.id}`,
    })
  }

  return {
    sources,
    selected,
    readySelected,
    usableIds,
    sourcesLoading: !sourcesSettled,
    sourcesFailed: sourcesSettled && sourcesResult.failed,
    retrySources: () => setSourcesAttempt(n => n + 1),
    chats,
    activeChatId,
    notes,
    cards,
    setCards,
    cardOrigins,
    uploading,
    sending,
    chatError,
    addSources,
    removeSource,
    toggleSource,
    toggleAllSources,
    send,
    selectChat,
    newChat,
    deleteChat,
    addNote,
    removeNote,
    addCard,
    addCardFromMessage,
  }
}
