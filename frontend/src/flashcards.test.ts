import { beforeEach, describe, expect, it } from 'vitest'

import { createCard, migrateLegacyFlashcards, reviewCard } from './flashcards'
import { readStored, writeStored } from './storage'

beforeEach(() => localStorage.clear())

describe('migrateLegacyFlashcards', () => {
  it('copies the active notebook deck into the single deck without removing the old key', () => {
    const cards = [createCard({ front: 'Q', back: 'A', origin: 'mcq:1' })]
    writeStored('active-notebook', 'nb1')
    writeStored('notebook:nb1:flashcards', cards)
    migrateLegacyFlashcards()
    expect(readStored('flashcards', [])).toEqual(cards)
    expect(readStored('notebook:nb1:flashcards', [])).toEqual(cards)
  })

  it('never overwrites an existing deck', () => {
    const mine = [createCard({ front: 'New', back: 'B' })]
    writeStored('flashcards', mine)
    writeStored('active-notebook', 'nb1')
    writeStored('notebook:nb1:flashcards', [createCard({ front: 'Old', back: 'B' })])
    migrateLegacyFlashcards()
    expect(readStored('flashcards', [])).toEqual(mine)
  })

  it('does nothing when there is no legacy data', () => {
    migrateLegacyFlashcards()
    expect(localStorage.length).toBe(0)
  })
})

describe('reviewCard', () => {
  it('moves remembered cards up a box and forgotten cards back to box 0', () => {
    const card = createCard({ front: 'Q', back: 'A' }, 0)
    const up = reviewCard(card, true, 0)
    expect(up.box).toBe(1)
    expect(up.due).toBe(86_400_000)
    expect(reviewCard(up, false, 0).box).toBe(0)
  })
})
