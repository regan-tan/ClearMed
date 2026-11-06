import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AppHeader } from './AppHeader'

const connection = vi.hoisted(() => ({ practicePreview: true }))
vi.mock('../api/client', () => connection)
afterEach(cleanup)

describe('AppHeader', () => {
  it.each([
    { mode: 'notebook' as const, preview: false, label: 'Frontend preview' },
    { mode: 'flashcards' as const, preview: false, label: 'Frontend preview' },
    { mode: 'practice' as const, preview: true, label: 'Practice preview' },
    { mode: 'practice' as const, preview: false, label: 'Practice backend' },
  ])('labels $mode with $label and switches modes', async ({ mode, preview, label }) => {
    connection.practicePreview = preview
    const onModeChange = vi.fn()
    render(
      <AppHeader
        notebooks={[{ id: 'notebook-1', title: 'Revision', createdAt: 0 }]}
        notebookId="notebook-1"
        onRenameNotebook={vi.fn()}
        onSelectNotebook={vi.fn()}
        onCreateNotebook={vi.fn()}
        onDeleteNotebook={vi.fn()}
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
    expect(screen.getByText(label)).toBeInTheDocument()
    if (mode === 'practice') expect(screen.queryByText('Frontend preview')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /practice/i }))
    expect(onModeChange).toHaveBeenCalledWith('practice')
  })
})
