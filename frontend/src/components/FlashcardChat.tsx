import { ArrowUp, Loader2, MessageSquare } from 'lucide-react'
import { useState } from 'react'
import { notebookApi, type NotebookMessage, type RetrievalSettings } from '../api/notebook'
import type { Flashcard } from '../flashcards'

const SUGGESTIONS = ['Explain this in more depth', 'Give me a clinical example', 'How would this be tested?']

interface Props {
  notebookId: string
  card: Flashcard
  retrieval: RetrievalSettings
}

// Follow-up chat about the card on screen, grounded in the card's source when it has one.
export function FlashcardChat({ notebookId, card, retrieval }: Props) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<NotebookMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function send(text: string) {
    const message = text.trim()
    if (!message || busy) return
    const history = messages.map(({ role, content }) => ({ role, content }))
    setMessages(current => [...current, { id: crypto.randomUUID(), role: 'user', content: message }])
    setInput('')
    setBusy(true)
    setError(null)
    try {
      const response = await notebookApi.chat(notebookId, {
        message,
        history,
        source_ids: card.sourceId ? [card.sourceId] : [],
        context: `Flashcard being revised.\nQuestion: ${card.front}\nAnswer: ${card.back}`,
        retrieval,
      })
      setMessages(current => [...current, { id: crypto.randomUUID(), role: 'assistant', ...response }])
    } catch {
      setMessages(current => current.slice(0, -1))
      setInput(message)
      setError('Couldn’t send your question. Please try again.')
    } finally {
      setBusy(false)
    }
  }

  if (!open)
    return (
      <button className="card-chat-toggle" onClick={() => setOpen(true)}>
        <MessageSquare size={15} />
        Ask about this card
      </button>
    )
  return (
    <div className="card-chat">
      {messages.length === 0 && (
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map(text => (
            <button key={text} className="card-chat-chip" onClick={() => send(text)}>
              {text}
            </button>
          ))}
        </div>
      )}
      {messages.map(message => (
        <div key={message.id} className={`card-chat-message ${message.role}`}>
          {message.content}
        </div>
      ))}
      {busy && (
        <p className="loading-message" role="status">
          <Loader2 className="animate-spin" size={15} />
          Thinking…
        </p>
      )}
      {error && (
        <p className="text-sm text-rose-600 dark:text-rose-400" role="alert">
          {error}
        </p>
      )}
      <form
        className="notebook-composer"
        onSubmit={e => {
          e.preventDefault()
          send(input)
        }}
      >
        <input
          autoFocus
          className="min-w-0 flex-1 bg-transparent py-1.5 text-sm outline-none placeholder:text-fg-subtle"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about this card…"
          aria-label="Ask about this card"
        />
        <button type="submit" aria-label="Send" disabled={!input.trim() || busy}>
          <ArrowUp size={18} />
        </button>
      </form>
    </div>
  )
}
