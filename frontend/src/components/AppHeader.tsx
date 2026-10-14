import { BookOpen, Moon, SlidersHorizontal, Sun } from 'lucide-react'
import { useCallback, useRef, useState, type ReactNode } from 'react'
import type { EmbeddingModel } from '../api/types'
import { useDismiss } from '../hooks/useDismiss'
import type { NotebookMeta } from '../notebooks'
import type { Theme } from '../theme'
import { NotebookSwitcher } from './NotebookSwitcher'
import { SearchSettings } from './SearchSettings'

export type Mode = 'notebook' | 'practice' | 'flashcards'

const MODES: { value: Mode; label: ReactNode }[] = [
  { value: 'notebook', label: 'Notebook' },
  {
    value: 'practice',
    label: (
      <>
        Practice<span className="mode-full"> MCQ</span>
      </>
    ),
  },
  { value: 'flashcards', label: 'Flashcards' },
]

interface Props {
  notebooks: NotebookMeta[]
  notebookId: string
  onRenameNotebook: (title: string) => void
  onSelectNotebook: (id: string) => void
  onCreateNotebook: () => void
  onDeleteNotebook: () => void
  mode: Mode
  onModeChange: (mode: Mode) => void
  model: EmbeddingModel
  topK: number
  onModelChange: (model: EmbeddingModel) => void
  onTopKChange: (topK: number) => void
  theme: Theme
  onToggleTheme: () => void
}

// Three columns: brand and notebook on the left, mode switch centred, settings and theme on the right.
export function AppHeader(props: Props) {
  const { mode, onModeChange, theme, onToggleTheme } = props
  const [settingsOpen, setSettingsOpen] = useState(false)
  const settingsRef = useRef<HTMLDivElement>(null)
  const closeSettings = useCallback(() => setSettingsOpen(false), [])
  useDismiss(settingsRef, settingsOpen, closeSettings)

  return (
    <header className="notebook-header">
      <div className="header-start">
        <a href="#chat" className="notebook-brand">
          <span className="brand-symbol">
            <BookOpen size={22} />
          </span>
          ClearMed
        </a>
        <span className="header-divider" />
        <NotebookSwitcher
          notebooks={props.notebooks}
          activeId={props.notebookId}
          onRename={props.onRenameNotebook}
          onSelect={props.onSelectNotebook}
          onCreate={props.onCreateNotebook}
          onDelete={props.onDeleteNotebook}
        />
      </div>

      <div className="mode-switch" role="group" aria-label="Mode">
        {MODES.map(({ value, label }) => (
          <button
            key={value}
            aria-pressed={mode === value}
            onClick={() => {
              onModeChange(value)
              closeSettings()
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="header-actions">
        <span className="preview-label">Frontend preview</span>
        <div className="settings-anchor" ref={settingsRef}>
          <button
            className="icon-button"
            onClick={() => setSettingsOpen(open => !open)}
            aria-label="Search settings"
            aria-expanded={settingsOpen}
            title="Search settings"
          >
            <SlidersHorizontal size={18} />
          </button>
          {settingsOpen && (
            <div className="settings-popover">
              <SearchSettings
                model={props.model}
                topK={props.topK}
                onModelChange={props.onModelChange}
                onTopKChange={props.onTopKChange}
                onClose={closeSettings}
              />
            </div>
          )}
        </div>
        <button
          className="icon-button"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  )
}
