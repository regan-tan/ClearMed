import { FileText, NotebookPen, Plus, X } from 'lucide-react'
import { useState } from 'react'
import type { Citation, Note } from '../api/notebook'
import { Markdown } from './Markdown'

interface Props {
  notes: Note[]
  /** When set, the panel shows this source passage instead of the notes. */
  citation: Citation | null
  mobileVisible: boolean
  onAddNote: (text: string) => void
  onRemoveNote: (id: string) => void
  onCloseCitation: () => void
  onClose: () => void
}

export function NotesPanel({
  notes,
  citation,
  mobileVisible,
  onAddNote,
  onRemoveNote,
  onCloseCitation,
  onClose,
}: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')

  function cancel() {
    setEditing(false)
    setDraft('')
  }

  return (
    <aside className={`notebook-panel notes-panel ${mobileVisible ? 'mobile-visible' : ''}`}>
      <div className="panel-heading">
        <h2>
          {citation ? (
            'Source passage'
          ) : (
            <>
              Notes <span className="count-badge">{notes.length}</span>
            </>
          )}
        </h2>
        <button
          className={`icon-button ${citation ? '' : 'mobile-close'}`}
          aria-label={citation ? 'Close source passage' : 'Close notes'}
          onClick={() => {
            onCloseCitation()
            onClose()
          }}
        >
          <X size={16} />
        </button>
      </div>

      {citation ? (
        <div className="citation-preview">
          <FileText size={22} />
          <h3>{citation.source_name}</h3>
          {citation.page && <p>Page {citation.page}</p>}
          <blockquote>{citation.excerpt}</blockquote>
          <button className="text-link" onClick={onCloseCitation}>
            Back to notes
          </button>
        </div>
      ) : (
        <>
          <div className="notes-intro">
            <p>Keep what clicks.</p>
            <span>Save useful explanations or add your own study notes.</span>
          </div>
          <button className="add-note" onClick={() => setEditing(true)}>
            <Plus size={16} />
            Write a note
          </button>
          {editing && (
            <form
              className="note-editor"
              onSubmit={e => {
                e.preventDefault()
                if (!draft.trim()) return
                onAddNote(draft.trim())
                cancel()
              }}
            >
              <textarea
                autoFocus
                aria-label="New study note"
                placeholder="What do you want to remember?"
                value={draft}
                onChange={e => setDraft(e.target.value)}
              />
              <div>
                <button type="button" onClick={cancel}>
                  Cancel
                </button>
                <button type="submit" disabled={!draft.trim()}>
                  Save note
                </button>
              </div>
            </form>
          )}
          <div className="notes-list">
            {notes.length
              ? notes.map(note => (
                  <article className="study-note" key={note.id}>
                    <div>
                      <NotebookPen size={15} />
                      <button className="icon-button" aria-label="Remove note" onClick={() => onRemoveNote(note.id)}>
                        <X size={13} />
                      </button>
                    </div>
                    <Markdown className="study-note-text">{note.text}</Markdown>
                  </article>
                ))
              : !editing && (
                  <div className="notes-empty">
                    <NotebookPen size={26} strokeWidth={1.2} />
                    <p>
                      A little space for
                      <br />
                      the important bits.
                    </p>
                  </div>
                )}
          </div>
          <p className="notes-footer">Saved in this browser. Clearing site data removes it.</p>
        </>
      )}
    </aside>
  )
}
