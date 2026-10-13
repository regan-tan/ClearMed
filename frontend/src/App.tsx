import { Stethoscope } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from './api/client'
import type { EmbeddingModel, Question } from './api/types'
import { ChatPanel } from './components/ChatPanel'
import { McqCard } from './components/McqCard'
import { QuestionList } from './components/QuestionList'
import { StudioPanel } from './components/StudioPanel'

export default function App() {
  const [questions, setQuestions] = useState<Question[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [model, setModel] = useState<EmbeddingModel>('e5')
  const [topK, setTopK] = useState(3)

  useEffect(() => {
    api.listQuestions().then((qs) => {
      setQuestions(qs)
      setActiveId(qs[0]?.id ?? null)
    })
  }, [])

  const active = questions.find((q) => q.id === activeId)

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-2 border-b border-slate-200 bg-white px-6 py-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white">
          <Stethoscope className="h-4 w-4" />
        </div>
        <span className="font-semibold">ClearMed Study</span>
        <span className="ml-2 hidden text-xs text-slate-400 sm:inline">Retrieval-grounded exam tutor</span>
      </header>

      <main className="grid flex-1 gap-4 overflow-y-auto p-4 lg:grid-cols-[280px_1fr_300px] lg:overflow-hidden">
        <div className="hidden min-h-0 lg:block">
          <QuestionList questions={questions} activeId={activeId} onSelect={setActiveId} />
        </div>

        <div className="flex min-h-0 flex-col gap-4 lg:overflow-y-auto">
          <select
            value={activeId ?? ''}
            onChange={(e) => setActiveId(e.target.value)}
            className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm lg:hidden"
          >
            {questions.map((q, i) => (
              <option key={q.id} value={q.id}>
                Q{i + 1} · {q.subject}
              </option>
            ))}
          </select>
          {active ? (
            // key resets MCQ and chat state when switching questions
            <div key={active.id} className="flex flex-col gap-4">
              <McqCard question={active} />
              <ChatPanel question={active} embeddingModel={model} topK={topK} />
            </div>
          ) : (
            <p className="p-6 text-sm text-slate-500">Loading questions…</p>
          )}
          <p className="pb-2 text-center text-[11px] text-slate-400">
            Educational research prototype. Not for diagnosis, treatment or clinical decision-making.
          </p>
        </div>

        <div className="min-h-0 overflow-y-auto">
          <StudioPanel model={model} topK={topK} onModelChange={setModel} onTopKChange={setTopK} />
        </div>
      </main>
    </div>
  )
}
