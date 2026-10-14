import { useEffect, type RefObject } from 'react'

/** Calls onDismiss on Escape or a pointer press outside `ref` while `open` is true (popovers and menus). */
export function useDismiss(ref: RefObject<HTMLElement | null>, open: boolean, onDismiss: () => void) {
  useEffect(() => {
    if (!open) return
    function onPointer(e: MouseEvent) {
      if (!ref.current?.contains(e.target as Node)) onDismiss()
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onDismiss()
    }
    document.addEventListener('mousedown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [ref, open, onDismiss])
}
