import { X } from 'lucide-react'
import type { EmbeddingModel } from '../api/types'
import { plural } from '../utils'

interface Props {
  model: EmbeddingModel
  topK: number
  onModelChange: (model: EmbeddingModel) => void
  onTopKChange: (topK: number) => void
  onClose: () => void
}

const MODELS: { value: EmbeddingModel; label: string }[] = [
  { value: 'minilm', label: 'MiniLM · General purpose' },
  { value: 'e5', label: 'E5-base-v2 · Retrieval' },
  { value: 'biolord', label: 'BioLORD-2023 · Biomedical' },
]

// Retrieval settings shared by notebook chat, flashcard questions and MCQ follow-ups.
export function SearchSettings({ model, topK, onModelChange, onTopKChange, onClose }: Props) {
  return (
    <section className="settings-panel" aria-label="Search settings">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">Search settings</h2>
          <p className="mt-1 text-xs text-fg-muted">
            How ClearMed finds supporting passages. Applies to notebook chat, flashcard questions and MCQ follow-ups.
          </p>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close settings">
          <X size={16} />
        </button>
      </div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <label className="text-xs font-medium">
          Search model
          <select
            className="settings-select"
            value={model}
            onChange={e => onModelChange(e.target.value as EmbeddingModel)}
          >
            {MODELS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs font-medium">
          Source passages
          <select className="settings-select" value={topK} onChange={e => onTopKChange(Number(e.target.value))}>
            {[1, 3, 5].map(k => (
              <option key={k} value={k}>
                {plural(k, 'passage')}
              </option>
            ))}
          </select>
        </label>
      </div>
    </section>
  )
}
