import clsx from 'clsx'
import { AlertTriangle, ArrowUp, Loader2, MessageCircle } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { ChatTurn, EmbeddingModel, Evidence, Question } from '../api/types'
import { EvidenceList } from './EvidenceList'

interface Message extends ChatTurn {
  evidence?: Evidence[]
  fallback?: boolean
}

interface Props {
  question: Question
  embeddingModel: EmbeddingModel
  topK: number
  // Content shown above the conversation in the same scroll area (the MCQ card).
  children?: ReactNode
}

const SUGGESTIONS = ['Explain the reasoning', 'Compare the options', 'Review the key concept']

export function ChatPanel({ question, embeddingModel, topK, children }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (messages.length === 0 && !loading) return
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

  async function send(text: string) {
    const message = text.trim()
    if (!message || loading) return
    const history: ChatTurn[] = messages.map(({ role, content }) => ({ role, content }))
    setMessages(m => [...m, { role: 'user', content: message }])
    setInput('')
    setLoading(true)
    try {
      const res = await api.followUp({
        question_id: question.id,
        history,
        message,
        embedding_model: embeddingModel,
        top_k: topK,
      })
      setMessages(m => [
        ...m,
        { role: 'assistant', content: res.reply, evidence: res.evidence, fallback: res.fallback },
      ])
    } catch (e) {
      setMessages(m => [
        ...m,
        { role: 'assistant', content: `Something went wrong (${e instanceof Error ? e.message : 'unknown error'}).` },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    // Keep the question, reasoning, and supporting evidence in one reading flow.
    <section className="study-card flex flex-col bg-surface">
      <div>
        <div className="mx-auto max-w-3xl px-5 py-7 sm:px-10 sm:py-9">
          {children}

          {messages.length > 0 && (
            <div className="mt-8 mb-5 flex items-center gap-3 text-xs font-medium text-fg-subtle">
              <span className="h-px flex-1 bg-border" />
              <MessageCircle className="h-3.5 w-3.5" /> Follow-up
              <span className="h-px flex-1 bg-border" />
            </div>
          )}

          <div className="space-y-4">
            {messages.map((m, i) =>
              m.role === 'user' ? (
                <div key={i} className="flex justify-end">
                  <p className="max-w-[80%] rounded-2xl rounded-br-md bg-bubble-user px-4 py-2.5 text-sm text-bubble-user-fg">
                    {m.content}
                  </p>
                </div>
              ) : (
                <div key={i} className="max-w-[90%]">
                  <div
                    className={clsx(
                      'rounded-2xl rounded-bl-md px-4 py-3 text-sm leading-relaxed',
                      m.fallback
                        ? 'border border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100'
                        : 'bg-subtle',
                    )}
                  >
                    {m.fallback && (
                      <p className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
                        <AlertTriangle className="h-3.5 w-3.5" /> Insufficient evidence
                      </p>
                    )}
                    {m.content}
                  </div>
                  {m.evidence && <EvidenceList evidence={m.evidence} />}
                </div>
              ),
            )}

            {loading && (
              <p className="flex items-center gap-2 text-sm text-fg-muted">
                <Loader2 className="h-4 w-4 animate-spin" /> Retrieving evidence…
              </p>
            )}
            <div ref={bottomRef} />
          </div>
        </div>
      </div>

      <div className="mx-auto w-full max-w-3xl px-5 pb-6 sm:px-10">
        <div className="mb-3 border-t border-border pt-5">
          <h2 className="text-sm font-semibold">Talk through the reasoning</h2>
          <p className="mt-1 text-xs text-fg-muted">Ask a follow-up and inspect the supporting passages.</p>
        </div>
        {messages.length === 0 && (
          <div className="mb-2.5 flex flex-wrap gap-2">
            {SUGGESTIONS.map(s => (
              <button
                key={s}
                onClick={() => send(s)}
                className="rounded-md border border-border px-3 py-1.5 text-xs text-fg-muted transition hover:border-primary hover:text-accent"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        <form
          onSubmit={e => {
            e.preventDefault()
            send(input)
          }}
          className="notebook-composer"
        >
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            aria-label="Follow-up question"
            placeholder="What would you like to understand?"
            className="min-w-0 flex-1 bg-transparent py-1.5 text-sm outline-none placeholder:text-fg-subtle"
          />
          <button type="submit" disabled={!input.trim() || loading} aria-label="Send">
            <ArrowUp className="h-[18px] w-[18px]" />
          </button>
        </form>
        <p className="mx-1 mt-2 text-[11px] text-fg-muted">
          Educational research prototype. Not for diagnosis, treatment or clinical decision-making.
        </p>
      </div>
    </section>
  )
}
