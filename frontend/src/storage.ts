import { useCallback, useEffect, useLayoutEffect, useRef, useState, type SetStateAction } from 'react'

const PREFIX = 'clearmed:v1:'

function sameShape(value: unknown, initial: unknown) {
  return value !== null && typeof value === typeof initial && Array.isArray(value) === Array.isArray(initial)
}

export function readStored<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    if (raw !== null) {
      const parsed: unknown = JSON.parse(raw)
      if (sameShape(parsed, fallback)) return parsed as T
    }
  } catch {
    // storage unavailable or value corrupt; fall back to the default
  }
  return fallback
}

export function writeStored(key: string, value: unknown) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    // quota exceeded or private mode; keep working in memory
  }
}

export function removeStored(key: string) {
  try { localStorage.removeItem(PREFIX + key) } catch { /* ignore */ }
}

export function moveStored(from: string, to: string) {
  try {
    const raw = localStorage.getItem(PREFIX + from)
    if (raw === null) return
    localStorage.setItem(PREFIX + to, raw)
    localStorage.removeItem(PREFIX + from)
  } catch { /* ignore */ }
}

export function hasStored(key: string) {
  try { return localStorage.getItem(PREFIX + key) !== null } catch { return false }
}

// useState backed by localStorage. This keeps the preview's data across refreshes in this browser;
// replace with backend calls once notebooks are stored server-side.
// The key may change (e.g. switching notebooks): the value is reloaded for the new key, and setters
// created under an earlier key keep writing to that key, so late async results land in the right place.
export function usePersistentState<T>(key: string, initial: T) {
  const [state, setState] = useState(() => ({ key, value: readStored(key, initial) }))
  let current = state
  if (state.key !== key) {
    current = { key, value: readStored(key, initial) }
    setState(current)
  }

  useEffect(() => { writeStored(current.key, current.value) }, [current.key, current.value])

  const activeKey = useRef(key)
  useLayoutEffect(() => { activeKey.current = key }, [key])
  const fallback = useRef(initial)

  const setValue = useCallback((action: SetStateAction<T>) => {
    const apply = (previous: T) => typeof action === 'function' ? (action as (value: T) => T)(previous) : action
    if (activeKey.current === key) setState(s => s.key === key ? { key, value: apply(s.value) } : s)
    else writeStored(key, apply(readStored(key, fallback.current)))
  }, [key])

  return [current.value, setValue] as const
}
