import { useState } from 'react'
import { ChevronDown, FileText } from 'lucide-react'
import type { Evidence } from '../api/types'

export function EvidenceList({ evidence }: { evidence: Evidence[] }) {
  const [open, setOpen] = useState(false)
  if (evidence.length === 0) return null

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        className="flex items-center gap-1 text-xs font-medium text-fg-muted hover:text-fg"
      >
        <FileText className="h-3.5 w-3.5" />
        {evidence.length} retrieved passage{evidence.length > 1 && 's'}
        <ChevronDown className={`h-3.5 w-3.5 transition ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <ol className="mt-2 space-y-2">
          {evidence.map((e, i) => (
            <li key={e.id} className="rounded-lg border border-border bg-surface p-3 text-xs">
              <div className="flex flex-wrap items-center gap-2 text-fg-muted">
                <span className="font-semibold text-fg">[{i + 1}]</span>
                <span>{e.subject}</span>
                <span className="font-mono text-fg-subtle">{e.id}</span>
                <span className="ml-auto font-mono">score {e.score.toFixed(3)}</span>
              </div>
              <p className="mt-1.5 leading-relaxed text-fg">{e.text}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
