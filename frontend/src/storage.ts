import { useEffect, useState } from 'react'

const PREFIX = 'clearmed:v1:'

function sameShape(value: unknown, initial: unknown) {
  return value !== null && typeof value === typeof initial && Array.isArray(value) === Array.isArray(initial)
}

// useState backed by localStorage. This keeps the preview's notebook across refreshes in this browser;
// replace with backend calls once notebooks are stored server-side.
export function usePersistentState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(PREFIX + key)
      if (raw !== null) {
        const parsed: unknown = JSON.parse(raw)
        if (sameShape(parsed, initial)) return parsed as T
      }
    } catch {
      // storage unavailable or value corrupt; fall back to the default
    }
    return initial
  })

  useEffect(() => {
    try {
      localStorage.setItem(PREFIX + key, JSON.stringify(value))
    } catch {
      // quota exceeded or private mode; keep working in memory
    }
  }, [key, value])

  return [value, setValue] as const
}
