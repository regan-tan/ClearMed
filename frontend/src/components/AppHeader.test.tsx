import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppHeader } from './AppHeader'

describe('AppHeader', () => {
  it('renders the preview label and switches modes', async () => {
    const onModeChange = vi.fn()
    render(
      <AppHeader
        notebooks={[{ id: 'notebook-1', title: 'Revision', createdAt: 0 }]}
        notebookId="notebook-1"
        onRenameNotebook={vi.fn()}
        onSelectNotebook={vi.fn()}
        onCreateNotebook={vi.fn()}
        onDeleteNotebook={vi.fn()}
        mode="notebook"
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
    expect(screen.getByText('Frontend preview')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /practice/i }))
    expect(onModeChange).toHaveBeenCalledWith('practice')
  })
})
