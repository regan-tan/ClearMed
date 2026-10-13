import clsx from 'clsx'
import { AlertTriangle, Loader2, MessageCircle, Send } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
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
}

const SUGGESTIONS = ['Why is this the right answer?', 'What happens if this goes wrong?', 'How is it treated?']

export function ChatPanel({ question, embeddingModel, topK }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function send(text: string) {
    const message = text.trim()
    if (!message || loading) return
    const history: ChatTurn[] = messages.map(({ role, content }) => ({ role, content }))
    setMessages((m) => [...m, { role: 'user', content: message }])
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
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: res.reply, evidence: res.evidence, fallback: res.fallback },
      ])
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: `Something went wrong (${e instanceof Error ? e.message : 'unknown error'}).` },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="flex min-h-[420px] flex-col rounded-2xl bg-white shadow-sm">
      <div className="flex items-center gap-2 border-b border-slate-100 px-5 py-3">
        <MessageCircle className="h-4 w-4 text-brand-600" />
        <h2 className="text-sm font-semibold">Ask a follow-up</h2>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {messages.length === 0 && (
          <div className="py-6 text-center">
            <p className="text-sm text-slate-500">Ask anything about this question. Answers are grounded in retrieved evidence.</p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:border-brand-500 hover:text-brand-700"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} className="flex justify-end">
              <p className="max-w-[80%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-sm text-white">{m.content}</p>
            </div>
          ) : (
            <div key={i} className="max-w-[90%]">
              <div
                className={clsx(
                  'rounded-2xl rounded-bl-md px-4 py-3 text-sm leading-relaxed',
                  m.fallback ? 'border border-amber-200 bg-amber-50 text-amber-900' : 'bg-slate-100',
                )}
              >
                {m.fallback && (
                  <p className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-amber-700">
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
          <p className="flex items-center gap-2 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" /> Retrieving evidence…
          </p>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
        className="flex gap-2 border-t border-slate-100 p-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. What does injury to this nerve cause?"
          className="flex-1 rounded-xl border border-slate-200 px-4 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
        />
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className="flex items-center justify-center rounded-xl bg-brand-600 px-4 text-white transition hover:bg-brand-700 disabled:opacity-40"
          aria-label="Send"
        >
          <Send className="h-4 w-4" />
        </button>
      </form>
    </section>
  )
}
