export interface Flashcard {
  id: string
  front: string
  back: string
  /** Short context shown on the card, e.g. the MCQ subject or "From chat". */
  label?: string
  /** Source the card was generated from; follow-up questions are grounded in it. */
  sourceId?: string
  /** Where the card was made from (e.g. `chat:<messageId>`), used to avoid duplicates. */
  origin?: string
  /** Leitner box: 0 is new or forgotten, higher boxes are reviewed less often. */
  box: number
  /** Timestamp (ms) when the card is next due. */
  due: number
  createdAt: number
}

export type FlashcardDraft = Pick<Flashcard, 'front' | 'back' | 'label' | 'origin' | 'sourceId'>

const DAY = 86_400_000
const INTERVAL_DAYS = [0, 1, 3, 7, 14, 30]

export function createCard(draft: FlashcardDraft, now = Date.now()): Flashcard {
  return { ...draft, id: crypto.randomUUID(), box: 0, due: now, createdAt: now }
}

// Remembered cards move up a box and come back later; forgotten cards return to box 0 and are due now.
export function reviewCard(card: Flashcard, remembered: boolean, now = Date.now()): Flashcard {
  const box = remembered ? Math.min(card.box + 1, INTERVAL_DAYS.length - 1) : 0
  return { ...card, box, due: now + INTERVAL_DAYS[box] * DAY }
}

export function nextInterval(card: Flashcard) {
  const days = INTERVAL_DAYS[Math.min(card.box + 1, INTERVAL_DAYS.length - 1)]
  return days === 1 ? '1 day' : `${days} days`
}

export function describeDue(due: number, now = Date.now()) {
  const days = Math.ceil((due - now) / DAY)
  if (days <= 0) return 'Due now'
  return days === 1 ? 'Due tomorrow' : `Due in ${days} days`
}
