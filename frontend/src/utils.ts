export function plural(count: number, word: string, pluralWord = `${word}s`) {
  return `${count} ${count === 1 ? word : pluralWord}`
}

export function relativeTime(time: number, now = Date.now()) {
  const minutes = Math.round((now - time) / 60_000)
  if (minutes < 1) return 'Just now'
  if (minutes < 60) return `${minutes} min ago`
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h ago`
  return new Date(time).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}
