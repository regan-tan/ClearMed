import clsx from 'clsx'
import { BookOpen } from 'lucide-react'
import type { Question } from '../api/types'

interface Props {
  questions: Question[]
  activeId: string | null
  onSelect: (id: string) => void
}

export function QuestionList({ questions, activeId, onSelect }: Props) {
  return (
    <aside className="flex h-full flex-col rounded-2xl bg-white shadow-sm">
      <div className="flex items-center gap-2 border-b border-slate-100 px-4 py-3">
        <BookOpen className="h-4 w-4 text-brand-600" />
        <h2 className="text-sm font-semibold">Question bank</h2>
        <span className="ml-auto text-xs text-slate-400">{questions.length}</span>
      </div>
      <ul className="flex-1 space-y-1 overflow-y-auto p-2">
        {questions.map((q, i) => (
          <li key={q.id}>
            <button
              onClick={() => onSelect(q.id)}
              className={clsx(
                'w-full rounded-xl px-3 py-2.5 text-left transition',
                q.id === activeId ? 'bg-brand-50 ring-1 ring-brand-100' : 'hover:bg-slate-50',
              )}
            >
              <span className="text-[11px] font-medium uppercase tracking-wide text-brand-600">
                {q.subject}
              </span>
              <p className="mt-0.5 line-clamp-2 text-sm text-slate-700">
                <span className="text-slate-400">Q{i + 1}. </span>
                {q.question}
              </p>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}
