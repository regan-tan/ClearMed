import type { KeyboardEvent, PointerEvent } from 'react'

interface Props {
  label: string
  /** Which panel this handle sizes: 'left' grows when dragged right, 'right' grows when dragged left. */
  side: 'left' | 'right'
  width: number
  min: number
  /** Upper bound, read at drag time so it can depend on the current window size. */
  getMax: () => number
  defaultWidth: number
  onChange: (width: number) => void
}

// Vertical drag handle between two notebook columns, VS Code style. Double-click resets the width.
export function ColumnResizer({ label, side, width, min, getMax, defaultWidth, onChange }: Props) {
  const direction = side === 'left' ? 1 : -1
  const clamp = (value: number) => Math.round(Math.min(Math.max(min, getMax()), Math.max(min, value)))

  function startResize(e: PointerEvent<HTMLDivElement>) {
    e.preventDefault()
    const handle = e.currentTarget
    try {
      handle.setPointerCapture(e.pointerId)
    } catch {
      /* pointer already released */
    }
    const startX = e.clientX
    const startWidth = width
    document.documentElement.classList.add('col-resizing')
    const move = (event: globalThis.PointerEvent) => onChange(clamp(startWidth + (event.clientX - startX) * direction))
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
    onChange(clamp(width + step * direction))
  }

  return (
    <div
      className={`column-resizer ${side}`}
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
