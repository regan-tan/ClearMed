import { ArrowUp, FileText, Layers, Loader2, MessageSquare, RotateCcw, Search, Sparkles, Trash2, Upload, X } from 'lucide-react'
import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { notebookApi, type NotebookMessage, type NotebookSource } from '../api/notebook'
import { createCard, describeDue, nextInterval, reviewCard, type Flashcard } from '../flashcards'

interface Props {
  cards: Flashcard[]
  setCards: Dispatch<SetStateAction<Flashcard[]>>
  sources: NotebookSource[]
  selected: string[]
  onToggleSource: (id: string) => void
  onAddSources: (files: File[]) => void
  onRemoveSource: (id: string) => void
  uploading: boolean
}

function dueIds(cards: Flashcard[], now = Date.now()) {
  return cards.filter(card => card.due <= now).sort((a, b) => a.due - b.due).map(card => card.id)
}

export function FlashcardsView({ cards, setCards, sources, selected, onToggleSource, onAddSources, onRemoveSource, uploading }: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const cardButton = useRef<HTMLButtonElement>(null)
  const [search, setSearch] = useState('')
  const [deckOpen, setDeckOpen] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [generateError, setGenerateError] = useState<string | null>(null)
  // Ids still to show in this sitting. Cram sessions rotate through cards without rescheduling them.
  const [queue, setQueue] = useState<string[]>(() => dueIds(cards))
  const [cram, setCram] = useState(false)
  const [flipped, setFlipped] = useState(false)
  const [reviewed, setReviewed] = useState(0)
  // Last deleted card, kept briefly so the delete can be undone.
  const [deleted, setDeleted] = useState<{ card: Flashcard; index: number; wasCurrent: boolean } | null>(null)

  const live = queue.filter(id => cards.some(card => card.id === id))
  const current = cards.find(card => card.id === live[0])
  const dueCount = dueIds(cards).length
  const filtered = cards.filter(card => `${card.front} ${card.back} ${card.label ?? ''}`.toLowerCase().includes(search.toLowerCase()))
  const nextDue = cards.length ? Math.min(...cards.map(card => card.due)) : null
  const chosen = selected.filter(id => sources.some(source => source.id === id))

  async function generate() {
    if (generating || !chosen.length) return
    setGenerating(true); setGenerateError(null)
    try {
      const generated = await notebookApi.generateFlashcards({ source_ids: chosen })
      const fresh = generated.map(item => createCard({ front: item.front, back: item.back, sourceId: item.source_id, label: sources.find(source => source.id === item.source_id)?.name ?? item.source_name }))
      setCards(all => [...fresh, ...all])
      if (!cram) setQueue(q => [...q, ...fresh.map(card => card.id)])
      setDeckOpen(false)
    } catch { setGenerateError('Couldn’t generate flashcards. Please try again.') }
    finally { setGenerating(false) }
  }
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
    setCram(all); setQueue(all ? cards.map(card => card.id) : dueIds(cards)); setReviewed(0); setFlipped(false)
  }
  function grade(remembered: boolean) {
    if (!current) return
    if (!cram) setCards(all => all.map(card => card.id === current.id ? reviewCard(card, remembered) : card))
    // Forgotten cards go to the back of this sitting so they come round again.
    setQueue(remembered ? live.slice(1) : [...live.slice(1), current.id])
    setReviewed(n => n + 1); setFlipped(false)
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (!current || target.closest('input, textarea, select') || e.metaKey || e.ctrlKey) return
      // Space/Enter keep their native meaning on other buttons; the card button flips itself when clicked.
      if (!flipped && (e.key === ' ' || e.key === 'Enter') && !target.closest('button')) { e.preventDefault(); setFlipped(true) }
      else if (flipped && e.key === '1') grade(false)
      else if (flipped && e.key === '2') grade(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // Move focus to each new card so Space flips it rather than re-pressing whatever was clicked last.
  useEffect(() => { cardButton.current?.focus({ preventScroll: true }) }, [current?.id])

  const total = reviewed + live.length

  return <>
    <div className="practice-tabs"><button onClick={() => setDeckOpen(open => !open)} aria-expanded={deckOpen}><Layers size={15} />Deck ({cards.length})</button></div>
    <main className="practice-layout">
      <div className={`bank-slot ${deckOpen ? 'mobile-visible' : ''}`}>
        <nav className="question-bank" aria-label="Flashcard deck">
          <div className="question-bank-heading"><h2 className="flex items-center gap-2 text-sm font-semibold">Deck <span className="count-badge">{cards.length}</span></h2><button onClick={() => setDeckOpen(false)} className="min-[901px]:hidden utility-button" aria-label="Close deck"><X size={16} /></button></div>

          <section className="generator" aria-label="Generate flashcards">
            <input ref={fileInput} type="file" multiple accept=".pdf,.docx,.txt,.md" className="hidden" onChange={e => { onAddSources(Array.from(e.target.files ?? [])); e.target.value = '' }} />
            {sources.length ? <>
              <p className="generator-title">Generate from your sources</p>
              <ul className="generator-sources">{sources.map(source => <li key={source.id}><label><input type="checkbox" checked={selected.includes(source.id)} onChange={() => onToggleSource(source.id)} /><FileText size={14} /><span title={source.name}>{source.name}</span></label><button className="icon-button delete-button" aria-label={`Delete ${source.name}`} title="Delete source" onClick={() => onRemoveSource(source.id)}><Trash2 size={14} /></button></li>)}</ul>
              <button className="generate-button" disabled={!chosen.length || generating} onClick={generate}>{generating ? <><Loader2 size={15} className="animate-spin" />Generating…</> : <><Sparkles size={15} />Generate flashcards</>}</button>
              <button className="text-link generator-upload" disabled={uploading} onClick={() => fileInput.current?.click()}>{uploading ? 'Adding…' : 'Upload more notes'}</button>
            </> : <>
              <p className="generator-title">Start with your material</p>
              <p className="generator-hint">Upload lecture notes or a textbook chapter and ClearMed will turn it into flashcards.</p>
              <button className="generate-button" disabled={uploading} onClick={() => fileInput.current?.click()}><Upload size={15} />{uploading ? 'Adding…' : 'Upload notes or textbook'}</button>
            </>}
            {generateError && <p className="generator-error" role="alert">{generateError}</p>}
          </section>

          {cards.length > 0 && <label className="question-search"><Search size={15} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search cards" aria-label="Search flashcards" /></label>}
          <ul className="question-items">{filtered.map(card => <li key={card.id} className={`deck-item ${card.id === current?.id ? 'active' : ''}`}>
            <div className="min-w-0 flex-1"><p className="line-clamp-2 text-[13px] font-medium leading-snug">{card.front}</p><p className="mt-1 truncate text-[11px] text-fg-muted">{card.label && <>{card.label} · </>}{describeDue(card.due)}</p></div>
            <div className="deck-actions"><button className="icon-button delete-button" aria-label={`Delete card: ${card.front}`} title="Delete card" onClick={() => deleteCard(card)}><Trash2 size={14} /></button></div>
          </li>)}</ul>
          {cards.length > 0 && !filtered.length && <p className="py-6 text-center text-sm text-fg-muted">No cards match.</p>}
        </nav>
      </div>

      <section className="study-card flashcard-stage bg-surface">
        {current ? <div className="flashcard-session">
          <div className="session-progress"><div className="session-meta"><span>{cram ? 'Studying all cards' : 'Review'} · {reviewed + 1} of {total}</span><button className="session-delete delete-button" onClick={() => deleteCard(current)}><Trash2 size={13} />Delete card</button></div><div className="progress-track"><span style={{ width: `${(reviewed / total) * 100}%` }} /></div></div>
          <button ref={cardButton} className={`flashcard ${flipped ? 'flipped' : ''}`} onClick={() => setFlipped(true)} disabled={flipped} aria-live="polite">
            {current.label && <span className="flashcard-label">{current.label}</span>}
            <p className="flashcard-front">{current.front}</p>
            {flipped ? <><hr /><p className="flashcard-back">{current.back}</p></> : <span className="flashcard-hint">Show answer <kbd>Space</kbd></span>}
          </button>
          {flipped && <div className="grade-buttons">
            <button onClick={() => grade(false)}><RotateCcw size={15} />Again<kbd>1</kbd></button>
            <button className="primary" onClick={() => grade(true)}>Got it{!cram && <small>· {nextInterval(current)}</small>}<kbd>2</kbd></button>
          </div>}
          {/* key resets the conversation when the next card comes up */}
          {flipped && <CardChat key={current.id} card={current} />}
        </div>
        : <div className="flashcard-empty">
          <div className="notebook-emblem"><Layers size={28} strokeWidth={1.4} /></div>
          {!cards.length ? <>
            <h2>Revise straight from your notes.</h2>
            <p>{sources.length ? 'Pick the sources to revise and ClearMed will write flashcards from them. Ask about any card while you study; cards you find hard come back sooner.' : 'Upload lecture notes or a textbook chapter and ClearMed will turn the key concepts into flashcards. Ask about any card while you study; cards you find hard come back sooner.'}</p>
            {sources.length
              ? <button className="primary-button" disabled={!chosen.length || generating} onClick={generate}><Sparkles size={15} />{generating ? 'Generating…' : `Generate from ${chosen.length} source${chosen.length !== 1 ? 's' : ''}`}</button>
              : <button className="primary-button" disabled={uploading} onClick={() => fileInput.current?.click()}><Upload size={15} />Upload notes or textbook</button>}
          </> : <>
            <h2>{reviewed ? 'Session complete.' : 'You’re all caught up.'}</h2>
            <p>{reviewed ? `You reviewed ${reviewed} card${reviewed !== 1 ? 's' : ''}. ` : ''}{dueCount ? `${dueCount} card${dueCount !== 1 ? 's are' : ' is'} due now.` : nextDue ? `Next card ${describeDue(nextDue).toLowerCase()}.` : ''}</p>
            <div className="flex flex-wrap items-center gap-3">
              {dueCount > 0 && <button className="primary-button" onClick={() => startSession(false)}>Review {dueCount} due</button>}
              <button className={dueCount ? 'text-link' : 'primary-button'} onClick={() => startSession(true)}>Study all {cards.length} cards</button>
            </div>
          </>}
        </div>}
      </section>
    </main>
    {deleted && <div className="toast" role="status">Card deleted<button onClick={undoDelete}>Undo</button></div>}
  </>
}

// Follow-up chat about the card on screen, grounded in the card's source when it has one.
function CardChat({ card }: { card: Flashcard }) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<NotebookMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function send(text: string) {
    const message = text.trim()
    if (!message || busy) return
    const history = messages.map(({ role, content }) => ({ role, content }))
    setMessages(current => [...current, { id: crypto.randomUUID(), role: 'user', content: message }]); setInput(''); setBusy(true); setError(null)
    try {
      const response = await notebookApi.chat({ message, history, source_ids: card.sourceId ? [card.sourceId] : [], context: `Flashcard being revised.\nQuestion: ${card.front}\nAnswer: ${card.back}` })
      setMessages(current => [...current, { id: crypto.randomUUID(), role: 'assistant', ...response }])
    } catch {
      setMessages(current => current.slice(0, -1)); setInput(message)
      setError('Couldn’t send your question. Please try again.')
    } finally { setBusy(false) }
  }

  if (!open) return <button className="card-chat-toggle" onClick={() => setOpen(true)}><MessageSquare size={15} />Ask about this card</button>
  return <div className="card-chat">
    {messages.length === 0 && <div className="flex flex-wrap gap-2">{['Explain this in more depth', 'Give me a clinical example', 'How would this be tested?'].map(text => <button key={text} className="card-chat-chip" onClick={() => send(text)}>{text}</button>)}</div>}
    {messages.map(message => <div key={message.id} className={`card-chat-message ${message.role}`}>{message.content}</div>)}
    {busy && <p className="loading-message" role="status"><Loader2 className="animate-spin" size={15} />Thinking…</p>}
    {error && <p className="text-sm text-rose-600 dark:text-rose-400" role="alert">{error}</p>}
    <form className="notebook-composer" onSubmit={e => { e.preventDefault(); send(input) }}>
      <input autoFocus className="min-w-0 flex-1 bg-transparent py-1.5 text-sm outline-none placeholder:text-fg-subtle" value={input} onChange={e => setInput(e.target.value)} placeholder="Ask about this card…" aria-label="Ask about this card" />
      <button type="submit" aria-label="Send" disabled={!input.trim() || busy}><ArrowUp size={18} /></button>
    </form>
  </div>
}
