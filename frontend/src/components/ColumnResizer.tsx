import type { KeyboardEvent, PointerEvent } from 'react'

interface Props {
  label: string
  width: number
  min: number
  /** Upper bound, read at drag time so it can depend on the current window size. */
  getMax: () => number
  defaultWidth: number
  onChange: (width: number) => void
}

// Vertical drag handle on the right edge of a left-hand panel. Dragging right widens the panel;
// arrow keys resize it too, and double-click resets the width.
export function ColumnResizer({ label, width, min, getMax, defaultWidth, onChange }: Props) {
  const clamp = (value: number) => Math.round(Math.min(Math.max(min, getMax()), Math.max(min, value)))

  function startResize(e: PointerEvent<HTMLDivElement>) {
    e.preventDefault()
    try {
      e.currentTarget.setPointerCapture(e.pointerId)
    } catch {
      /* pointer already released */
    }
    const startX = e.clientX
    const startWidth = width
    document.documentElement.classList.add('col-resizing')
    const move = (event: globalThis.PointerEvent) => onChange(clamp(startWidth + (event.clientX - startX)))
    const stop = () => {
      document.documentElement.classList.remove('col-resizing')
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', stop)
      window.removeEventListener('pointercancel', stop)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', stop)
    window.addEventListener('pointercancel', stop)
  }

  function resizeWithKeys(e: KeyboardEvent<HTMLDivElement>) {
    const step = e.key === 'ArrowLeft' ? -16 : e.key === 'ArrowRight' ? 16 : 0
    if (!step) return
    e.preventDefault()
    onChange(clamp(width + step))
  }

  return (
    <div
      className="column-resizer"
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuemin={min}
      aria-valuenow={width}
      tabIndex={0}
      onPointerDown={startResize}
      onKeyDown={resizeWithKeys}
      onDoubleClick={() => onChange(clamp(defaultWidth))}
    />
  )
}
