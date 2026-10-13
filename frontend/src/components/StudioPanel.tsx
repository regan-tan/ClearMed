import clsx from 'clsx'
import { Layers, ListChecks, NotebookPen, SlidersHorizontal } from 'lucide-react'
import type { EmbeddingModel } from '../api/types'

const MODELS: { id: EmbeddingModel; name: string; hint: string }[] = [
  { id: 'minilm', name: 'MiniLM', hint: 'General-purpose' },
  { id: 'e5', name: 'E5-base-v2', hint: 'Retrieval-specialised' },
  { id: 'biolord', name: 'BioLORD-2023', hint: 'Biomedical' },
]

const DEPTHS = [1, 3, 5]

const UPCOMING = [
  { icon: Layers, name: 'Flashcards', hint: 'Review key facts from this topic' },
  { icon: ListChecks, name: 'Practice quiz', hint: 'More questions on the same subject' },
  { icon: NotebookPen, name: 'Study notes', hint: 'Summary of your session' },
]

interface Props {
  model: EmbeddingModel
  topK: number
  onModelChange: (m: EmbeddingModel) => void
  onTopKChange: (k: number) => void
}

export function StudioPanel({ model, topK, onModelChange, onTopKChange }: Props) {
  return (
    <aside className="flex h-full flex-col gap-4">
      <div className="rounded-2xl bg-white p-4 shadow-sm">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-brand-600" />
          <h2 className="text-sm font-semibold">Retrieval settings</h2>
        </div>

        <p className="mt-4 text-xs font-medium text-slate-500">Embedding model</p>
        <div className="mt-2 space-y-1.5">
          {MODELS.map((m) => (
            <button
              key={m.id}
              onClick={() => onModelChange(m.id)}
              className={clsx(
                'flex w-full items-center justify-between rounded-xl border px-3 py-2 text-left text-sm transition',
                m.id === model ? 'border-brand-500 bg-brand-50' : 'border-slate-200 hover:bg-slate-50',
              )}
            >
              <span className="font-medium">{m.name}</span>
              <span className="text-xs text-slate-500">{m.hint}</span>
            </button>
          ))}
        </div>

        <p className="mt-4 text-xs font-medium text-slate-500">Passages retrieved (top-k)</p>
        <div className="mt-2 grid grid-cols-3 gap-1.5">
          {DEPTHS.map((k) => (
            <button
              key={k}
              onClick={() => onTopKChange(k)}
              className={clsx(
                'rounded-lg border py-1.5 text-sm font-medium transition',
                k === topK ? 'border-brand-500 bg-brand-50 text-brand-700' : 'border-slate-200 hover:bg-slate-50',
              )}
            >
              k = {k}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-2xl bg-white p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Study tools</h2>
        <div className="mt-3 space-y-2">
          {UPCOMING.map(({ icon: Icon, name, hint }) => (
            <div key={name} className="flex items-start gap-3 rounded-xl border border-dashed border-slate-200 p-3">
              <Icon className="mt-0.5 h-4 w-4 text-slate-400" />
              <div>
                <p className="text-sm font-medium text-slate-600">
                  {name} <span className="ml-1 text-[10px] font-normal uppercase text-slate-400">Soon</span>
                </p>
                <p className="text-xs text-slate-400">{hint}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  )
}
