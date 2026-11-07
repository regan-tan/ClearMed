import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SearchSettings } from './SearchSettings'

afterEach(cleanup)

const base = { model: 'minilm' as const, topK: 3, onModelChange: vi.fn(), onTopKChange: vi.fn(), onClose: vi.fn() }
const models = [
  { id: 'a', label: 'Model A' },
  { id: 'b', label: 'Model B' },
]

describe('SearchSettings answer model', () => {
  it.each([[undefined], [[]], [[models[0]]]])('is hidden when there is no choice: %j', list => {
    render(<SearchSettings {...base} llmModels={list} llm="a" onLlmChange={vi.fn()} />)
    expect(screen.queryByLabelText('Answer model')).not.toBeInTheDocument()
  })

  it('offers every model the backend lists and reports the choice', async () => {
    const onLlmChange = vi.fn()
    render(<SearchSettings {...base} llmModels={models} llm="a" onLlmChange={onLlmChange} />)
    const select = screen.getByLabelText('Answer model')
    expect(select).toHaveValue('a')
    await userEvent.selectOptions(select, 'b')
    expect(onLlmChange).toHaveBeenCalledWith('b')
  })
})
