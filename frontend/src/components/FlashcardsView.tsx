import clsx from 'clsx'
import { Layers, RotateCcw, RotateCw, Search, Trash2, X } from 'lucide-react'
import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { BANK_DEFAULT, BANK_MIN, bankMax } from '../bankWidth'
import { describeDue, nextInterval, reviewCard, type Flashcard } from '../flashcards'
import { plural } from '../utils'
import { ColumnResizer } from './ColumnResizer'
import { Markdown } from './Markdown'

interface Props {
  cards: Flashcard[]
  setCards: Dispatch<SetStateAction<Flashcard[]>>
  bankWidth: number
  onBankWidth: (width: number) => void
}

const CARD_EXIT_MS = 260

function sessionSummary(reviewed: number, dueCount: number, nextDue: number | null) {
  const done = reviewed ? `You reviewed ${plural(reviewed, 'card')}. ` : ''
  if (dueCount) return `${done}${plural(dueCount, 'card is', 'cards are')} due now.`
  return nextDue ? `${done}Next card ${describeDue(nextDue).toLowerCase()}.` : done
}

function dueIds(cards: Flashcard[], now = Date.now()) {
  return cards
    .filter(card => card.due <= now)
    .sort((a, b) => a.due - b.due)
    .map(card => card.id)
}

export function FlashcardsView({ cards, setCards, bankWidth, onBankWidth }: Props) {
  const cardButton = useRef<HTMLButtonElement>(null)
  const [search, setSearch] = useState('')
  const [deckOpen, setDeckOpen] = useState(false)
  // Ids still to show in this sitting. Cram sessions rotate through cards without rescheduling them.
  const [queue, setQueue] = useState<string[]>(() => dueIds(cards))
  const [cram, setCram] = useState(false)
  const [flipped, setFlipped] = useState(false)
  // Direction the graded card slides out: left for "Again", right for "Got it".
  const [leaving, setLeaving] = useState<'left' | 'right' | null>(null)
  const [reviewed, setReviewed] = useState(0)
  // Last deleted card, kept briefly so the delete can be undone.
  const [deleted, setDeleted] = useState<{ card: Flashcard; index: number; wasCurrent: boolean } | null>(null)

  const live = queue.filter(id => cards.some(card => card.id === id))
  const current = cards.find(card => card.id === live[0])
  const dueCount = dueIds(cards).length
  const filtered = cards.filter(card =>
    `${card.front} ${card.back} ${card.label ?? ''}`.toLowerCase().includes(search.toLowerCase()),
  )
  const nextDue = cards.length ? Math.min(...cards.map(card => card.due)) : null

  function deleteCard(card: Flashcard) {
    const wasCurrent = card.id === current?.id
    setDeleted({ card, index: cards.findIndex(item => item.id === card.id), wasCurrent })
    setCards(all => all.filter(item => item.id !== card.id))
    if (wasCurrent) setFlipped(false)
  }
  function undoDelete() {
    if (!deleted) return
    const { card, index, wasCurrent } = deleted
    setCards(all => [...all.slice(0, index), card, ...all.slice(index)])
    if (wasCurrent) setQueue(q => [card.id, ...q.filter(id => id !== card.id)])
    setDeleted(null)
  }
  useEffect(() => {
    if (!deleted) return
    const timer = setTimeout(() => setDeleted(null), 6000)
    return () => clearTimeout(timer)
  }, [deleted])
  function startSession(all: boolean) {
    setCram(all)
    setQueue(all ? cards.map(card => card.id) : dueIds(cards))
    setReviewed(0)
    setFlipped(false)
  }
  function grade(remembered: boolean) {
    if (!current || leaving) return
    const card = current
    const finish = () => {
      if (!cram) setCards(all => all.map(item => (item.id === card.id ? reviewCard(item, remembered) : item)))
      // Forgotten cards go to the back of this sitting so they come round again.
      setQueue(remembered ? live.slice(1) : [...live.slice(1), card.id])
      setReviewed(n => n + 1)
      setFlipped(false)
      setLeaving(null)
    }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return finish()
    setLeaving(remembered ? 'right' : 'left')
    setTimeout(finish, CARD_EXIT_MS)
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (!current || target.closest('input, textarea, select') || e.metaKey || e.ctrlKey) return
      // Space/Enter keep their native meaning on other buttons; the card button flips itself when clicked.
      if ((e.key === ' ' || e.key === 'Enter') && !target.closest('button')) {
        e.preventDefault()
        if (!leaving) setFlipped(value => !value)
      } else if (flipped && e.key === '1') grade(false)
      else if (flipped && e.key === '2') grade(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // Move focus to each new card so Space flips it rather than re-pressing whatever was clicked last.
  useEffect(() => {
    cardButton.current?.focus({ preventScroll: true })
  }, [current?.id])

  const total = reviewed + live.length

  return (
    <>
      <div className="practice-tabs">
        <button onClick={() => setDeckOpen(open => !open)} aria-expanded={deckOpen}>
          <Layers size={15} />
          Deck ({cards.length})
        </button>
      </div>
      <main className="practice-layout">
        <div className={`bank-slot ${deckOpen ? 'mobile-visible' : ''}`}>
          <nav className="question-bank" aria-label="Flashcard deck">
            <div className="question-bank-heading">
              <h2 className="flex items-center gap-2 text-sm font-semibold">
                Deck <span className="count-badge">{cards.length}</span>
              </h2>
              <button
                onClick={() => setDeckOpen(false)}
                className="icon-button min-[901px]:hidden"
                aria-label="Close deck"
              >
                <X size={16} />
              </button>
            </div>

            {cards.length > 0 && (
              <label className="question-search">
                <Search size={15} />
                <input
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search cards"
                  aria-label="Search flashcards"
                />
              </label>
            )}
            <ul className="question-items">
              {filtered.map(card => (
                <li key={card.id} className={`deck-item ${card.id === current?.id ? 'active' : ''}`}>
                  <div className="min-w-0 flex-1">
                    <p className="line-clamp-2 text-[13px] font-medium leading-snug">{card.front}</p>
                    <p className="mt-1 truncate text-[11px] text-fg-muted">
                      {card.label && <>{card.label} · </>}
                      {describeDue(card.due)}
                    </p>
                  </div>
                  <div className="deck-actions">
                    <button
                      className="icon-button delete-button"
                      aria-label={`Delete card: ${card.front}`}
                      title="Delete card"
                      onClick={() => deleteCard(card)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </li>
              ))}
            </ul>
            {cards.length > 0 && !filtered.length && (
              <p className="py-6 text-center text-sm text-fg-muted">No cards match.</p>
            )}
          </nav>
          <ColumnResizer
            label="Resize deck"
            width={bankWidth}
            min={BANK_MIN}
            getMax={bankMax}
            defaultWidth={BANK_DEFAULT}
            onChange={onBankWidth}
          />
        </div>

        <section className="study-card flashcard-stage bg-surface">
          {current ? (
            <div className="flashcard-session">
              <div className="session-progress">
                <div className="session-meta">
                  <span>
                    {cram ? 'Studying all cards' : 'Review'} · {reviewed + 1} of {total}
                  </span>
                  <button className="session-delete delete-button" onClick={() => deleteCard(current)}>
                    <Trash2 size={13} />
                    Delete card
                  </button>
                </div>
                <div className="progress-track">
                  <span style={{ width: `${(reviewed / total) * 100}%` }} />
                </div>
              </div>
              {/* Cards still to come show as a stack behind the current one. */}
              <div className={`flashcard-deck stack-${Math.min(live.length - 1, 2)}`}>
                {/* Remount per card and review so a new card starts face up instead of flipping back into view. */}
                <button
                  key={`${current.id}-${reviewed}`}
                  ref={cardButton}
                  className={clsx('flashcard', flipped && 'flipped', leaving && `leaving-${leaving}`)}
                  onClick={() => !leaving && setFlipped(value => !value)}
                  aria-live="polite"
                >
                  <span className="flashcard-inner">
                    <span className="flashcard-face front" aria-hidden={flipped}>
                      <span className="flashcard-top">
                        <span className="flashcard-eyebrow">Question</span>
                        {current.label && <span className="flashcard-label">{current.label}</span>}
                      </span>
                      <span className="flashcard-front">{current.front}</span>
                      <span className="flashcard-hint">
                        <RotateCw size={14} />
                        Tap to reveal the answer <kbd>Space</kbd>
                      </span>
                    </span>
                    <span className="flashcard-face back" aria-hidden={!flipped}>
                      <span className="flashcard-top">
                        <span className="flashcard-eyebrow">Answer</span>
                        {current.label && <span className="flashcard-label">{current.label}</span>}
                      </span>
                      <span className="flashcard-question">{current.front}</span>
                      <Markdown className="flashcard-back" links={false}>
                        {current.back}
                      </Markdown>
                      <span className="flashcard-hint">
                        <RotateCw size={14} />
                        Tap to see the question
                      </span>
                    </span>
                  </span>
                </button>
              </div>
              <div className="grade-buttons">
                <button className="again" disabled={!flipped || !!leaving} onClick={() => grade(false)}>
                  <RotateCcw size={15} />
                  Again<kbd>1</kbd>
                </button>
                <button className="primary" disabled={!flipped || !!leaving} onClick={() => grade(true)}>
                  Got it{!cram && <small>· {nextInterval(current)}</small>}
                  <kbd>2</kbd>
                </button>
              </div>
            </div>
          ) : (
            <div className="flashcard-empty">
              <div className="notebook-emblem">
                <Layers size={28} strokeWidth={1.4} />
              </div>
              {!cards.length ? (
                <>
                  <h2>No flashcards yet.</h2>
                  <p>
                    Answer a practice question, then choose “Add to flashcards” to keep it here. Cards you find hard
                    come back sooner.
                  </p>
                </>
              ) : (
                <>
                  <h2>{reviewed ? 'Session complete.' : 'You’re all caught up.'}</h2>
                  <p>{sessionSummary(reviewed, dueCount, nextDue)}</p>
                  <div className="flex flex-wrap items-center gap-3">
                    {dueCount > 0 && (
                      <button className="primary-button" onClick={() => startSession(false)}>
                        Review {dueCount} due
                      </button>
                    )}
                    <button className={dueCount ? 'text-link' : 'primary-button'} onClick={() => startSession(true)}>
                      Study all {cards.length} cards
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </section>
      </main>
      {deleted && (
        <div className="toast" role="status">
          Card deleted<button onClick={undoDelete}>Undo</button>
        </div>
      )}
    </>
  )
}
