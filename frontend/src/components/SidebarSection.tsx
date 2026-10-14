import { ChevronRight } from 'lucide-react'
import type { CSSProperties, ReactNode } from 'react'

interface Props {
  title: string
  count?: number
  open: boolean
  onToggle: () => void
  /** Icon buttons shown on the right of the header, e.g. "add" or "new". */
  actions?: ReactNode
  style?: CSSProperties
  children: ReactNode
}

// Collapsible sidebar view in the style of VS Code's Explorer sections.
export function SidebarSection({ title, count, open, onToggle, actions, style, children }: Props) {
  return <section className={`sidebar-section ${open ? 'open' : ''}`} style={style}>
    <div className="sidebar-section-header">
      <button className="sidebar-section-toggle" onClick={onToggle} aria-expanded={open}>
        <ChevronRight size={14} className="sidebar-chevron" />{title}{count !== undefined && <span className="count-badge">{count}</span>}
      </button>
      {actions && <div className="sidebar-section-actions">{actions}</div>}
    </div>
    {open && <div className="sidebar-section-body">{children}</div>}
  </section>
}
