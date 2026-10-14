import { Check, Search, X } from 'lucide-react'
import { useState } from 'react'
import type { OptionKey, Question } from '../api/types'

interface Props {
  questions: Question[]
  activeId: string | null
  answers: Record<string, OptionKey>
  onSelect: (id: string) => void
  onCollapse: () => void
}

export function QuestionList({ questions, activeId, answers, onSelect, onCollapse }: Props) {
  const [search, setSearch] = useState('')
  const filtered = questions.filter(q => `${q.subject} ${q.topic ?? ''} ${q.question}`.toLowerCase().includes(search.toLowerCase()))
  return <nav className="question-bank" aria-label="Question bank">
    <div className="question-bank-heading"><h2 className="flex items-center gap-2 text-sm font-semibold">Question bank <span className="count-badge">{questions.length}</span></h2><button onClick={onCollapse} className="min-[901px]:hidden utility-button" aria-label="Close question bank"><X size={16} /></button></div>
    <label className="question-search"><Search size={15} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Find a topic or question" aria-label="Search questions" /></label>
    <ul className="question-items">{filtered.map(q => {
      const answered = answers[q.id] !== undefined
      return <li key={q.id}><button className={`question-link ${q.id === activeId ? 'active' : ''}`} aria-current={q.id === activeId ? 'step' : undefined} onClick={() => onSelect(q.id)}><span className={`question-number ${answered ? 'completed' : ''}`}>{answered ? <Check size={14} /> : String(questions.indexOf(q) + 1).padStart(2, '0')}</span><span className="min-w-0"><span className="block text-xs font-semibold mb-1">{q.subject}</span><span className="line-clamp-2 text-xs leading-relaxed text-fg-muted">{q.question}</span>{answered && <span className="block mt-1 text-[11px] text-accent">{answers[q.id] === q.correct ? 'Correct' : 'Review answer'}</span>}</span></button></li>
    })}</ul>
    {!filtered.length && <p className="py-6 text-sm text-fg-muted">No matches. Try another topic.</p>}
  </nav>
}
