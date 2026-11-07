import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ColumnResizer } from './ColumnResizer'

afterEach(cleanup)

function setup(width = 280) {
  const onChange = vi.fn()
  render(
    <ColumnResizer
      label="Resize list"
      width={width}
      min={200}
      getMax={() => 500}
      defaultWidth={280}
      onChange={onChange}
    />,
  )
  return { handle: screen.getByRole('separator', { name: 'Resize list' }), onChange }
}

describe('ColumnResizer', () => {
  it('widens and narrows with the arrow keys, clamped to the limits', () => {
    const { handle, onChange } = setup(490)
    fireEvent.keyDown(handle, { key: 'ArrowRight' })
    expect(onChange).toHaveBeenLastCalledWith(500)
    fireEvent.keyDown(handle, { key: 'ArrowRight' })
    expect(onChange).toHaveBeenLastCalledWith(500) // Never past the maximum.
    cleanup()
    const narrow = setup(205)
    fireEvent.keyDown(narrow.handle, { key: 'ArrowLeft' })
    expect(narrow.onChange).toHaveBeenLastCalledWith(200) // Never below the minimum.
  })

  it('ignores other keys', () => {
    const { handle, onChange } = setup()
    fireEvent.keyDown(handle, { key: 'a' })
    expect(onChange).not.toHaveBeenCalled()
  })

  it('resets to the default width on double-click', async () => {
    const { handle, onChange } = setup(420)
    await userEvent.dblClick(handle)
    expect(onChange).toHaveBeenLastCalledWith(280)
  })

  it('follows a pointer drag and stops when released', () => {
    const { handle, onChange } = setup(300)
    fireEvent.pointerDown(handle, { clientX: 100, pointerId: 1 })
    expect(document.documentElement).toHaveClass('col-resizing')
    fireEvent(window, new MouseEvent('pointermove', { clientX: 160 }))
    expect(onChange).toHaveBeenLastCalledWith(360)
    fireEvent(window, new MouseEvent('pointermove', { clientX: 900 }))
    expect(onChange).toHaveBeenLastCalledWith(500)
    fireEvent(window, new MouseEvent('pointerup'))
    expect(document.documentElement).not.toHaveClass('col-resizing')
    onChange.mockClear()
    fireEvent(window, new MouseEvent('pointermove', { clientX: 200 }))
    expect(onChange).not.toHaveBeenCalled()
  })
})
