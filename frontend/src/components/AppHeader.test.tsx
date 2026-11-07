import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AppHeader } from './AppHeader'

afterEach(cleanup)

describe('AppHeader', () => {
  it.each(['practice', 'flashcards'] as const)('shows %s as the pressed mode and switches modes', async mode => {
    const onModeChange = vi.fn()
    render(
      <AppHeader
        mode={mode}
        onModeChange={onModeChange}
        model="e5"
        topK={3}
        onModelChange={vi.fn()}
        onTopKChange={vi.fn()}
        theme="light"
        onToggleTheme={vi.fn()}
      />,
    )

    expect(screen.getByText('ClearMed')).toBeInTheDocument()
    expect(screen.queryByText(/preview/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Notebook')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /practice/i })).toHaveAttribute('aria-pressed', String(mode === 'practice'))
    const other = mode === 'practice' ? /flashcards/i : /practice/i
    await userEvent.click(screen.getByRole('button', { name: other }))
    expect(onModeChange).toHaveBeenCalledWith(mode === 'practice' ? 'flashcards' : 'practice')
  })
})
