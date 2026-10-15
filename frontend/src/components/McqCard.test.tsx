import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { McqCard } from './McqCard'

describe('McqCard', () => {
  it('allows an answer to be selected and submitted', async () => {
    const onAnswer = vi.fn()
    render(
      <McqCard
        question={{
          id: 'q1',
          question: 'Which option is correct?',
          options: { A: 'First', B: 'Second', C: 'Third', D: 'Fourth' },
          correct: 'B',
          subject: 'Test',
          topic: 'Topic',
        }}
        onAnswer={onAnswer}
        inDeck={false}
        onFlashcard={vi.fn()}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: /second/i }))
    expect(screen.getByText('Option B selected')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Check answer' }))
    expect(onAnswer).toHaveBeenCalledWith('B')
  })
})
