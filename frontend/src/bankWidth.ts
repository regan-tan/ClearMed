// Width of the left panel (question list / flashcard deck). The user can drag it wider or narrower.
export const BANK_DEFAULT = 280
export const BANK_MIN = 200
const BANK_MAX = 720
const MIN_MAIN = 420 // The study area keeps at least this much room.

export function bankMax() {
  return Math.max(BANK_MIN, Math.min(BANK_MAX, window.innerWidth - MIN_MAIN))
}
