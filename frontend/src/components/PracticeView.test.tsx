import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Question } from '../api/types'
import { PracticeView } from './PracticeView'
import contract from '../../../tests/fixtures/demo_api.json'

vi.hoisted(() => vi.stubEnv('VITE_USE_MOCK', 'false'))

const first: Question = {
  id: 'first', question: 'First question?', subject: 'Test', correct: 'B',
  options: { A: 'Alpha', B: 'Beta', C: 'Gamma', D: 'Delta' },
}
const second: Question = { ...first, id: 'second', question: 'Second question?' }
const fetchMock = vi.fn()
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
})
const prediction = (id = 'first', explanation = 'A model explanation.') =>
  json({ question_id: id, predicted: 'A', explanation })

function show() {
  return render(<PracticeView model="minilm" topK={3} flashcardOrigins={[]} onFlashcard={vi.fn()} bankWidth={280} onBankWidth={vi.fn()} />)
}

async function answer() {
  await screen.findByRole('heading', { name: 'First question?' })
  await userEvent.click(screen.getByRole('button', { name: /Beta/ }))
  await userEvent.click(screen.getByRole('button', { name: 'Check answer' }))
}

beforeEach(() => {
  localStorage.clear()
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Practice MCQ with the HTTP adapter', () => {
  it('completes question, prediction and follow-up using the shared FastAPI contract', async () => {
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
    fetchMock.mockResolvedValueOnce(json(contract.questions))
      .mockResolvedValueOnce(json(contract.prediction))
      .mockResolvedValueOnce(json(contract.follow_up))
    render(<PracticeView model="minilm" topK={1} flashcardOrigins={[]} onFlashcard={vi.fn()} bankWidth={280} onBankWidth={vi.fn()} />)
    await screen.findByRole('heading', { name: contract.questions[0].question })
    await userEvent.click(screen.getByRole('button', { name: /propranolol/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Check answer' }))
    await screen.findByText(contract.prediction.explanation)
    expect(screen.getByText('(differs from the answer key)')).toBeInTheDocument()
    await userEvent.type(screen.getByRole('textbox', { name: 'Follow-up question' }), contract.follow_up_request.message)
    await userEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByText(/stub reply using 3 evidence passage/)).toBeInTheDocument()
    expect(JSON.parse(fetchMock.mock.calls[2][1].body)).toEqual(contract.follow_up_request)
    await userEvent.click(screen.getByRole('button', { name: '1 retrieved passage' }))
    expect(screen.getByText(contract.follow_up.evidence[0].text)).toBeInTheDocument()
    expect(screen.getByText(contract.follow_up.evidence[0].id)).toBeInTheDocument()
  })

  it('loads questions, submits only the ID, and distinguishes the model from the answer key', async () => {
    let finish!: (response: Response) => void
    fetchMock.mockResolvedValueOnce(json([first]))
      .mockImplementationOnce(() => new Promise<Response>(resolve => { finish = resolve }))
    show()
    expect(screen.getByText('Loading questions…')).toBeInTheDocument()
    await screen.findByRole('heading', { name: 'First question?' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(screen.queryByText('Correct.')).not.toBeInTheDocument()
    await answer()
    expect(screen.getByText('Correct.')).toBeInTheDocument()
    expect(screen.getByText('Generating…')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenLastCalledWith('/api/questions/first/predict', {
      headers: { 'Content-Type': 'application/json' }, method: 'POST',
    })
    await act(async () => finish(prediction()))
    expect(await screen.findByText('A model explanation.')).toBeInTheDocument()
    expect(screen.getByText('(differs from the answer key)')).toBeInTheDocument()
    expect(JSON.parse(localStorage.getItem('clearmed:v1:practice-answers')!)).toEqual({ first: 'B' })
  })

  it('recovers from an unavailable question bank', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Network error')).mockResolvedValueOnce(json([first]))
    show()
    expect(await screen.findByText(/Cannot reach the tutor backend/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('heading', { name: 'First question?' })).toBeInTheDocument()
  })

  it('shows an empty bank instead of loading forever and allows retry', async () => {
    fetchMock.mockResolvedValueOnce(json([])).mockResolvedValueOnce(json([first]))
    show()
    expect(await screen.findByText('No practice questions are available.')).toBeInTheDocument()
    expect(screen.queryByText('Loading questions…')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('heading', { name: 'First question?' })).toBeInTheDocument()
  })

  it('preserves the saved answer when prediction fails and retries the explanation', async () => {
    fetchMock.mockResolvedValueOnce(json([first]))
      .mockResolvedValueOnce(json({ detail: 'The MCQ model gave no valid A-D answer (MALFORMED).' }, 502))
      .mockResolvedValueOnce(prediction('first', ''))
    show()
    await answer()
    expect(await screen.findByText(/MALFORMED/)).toBeInTheDocument()
    expect(screen.getByText('Correct.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByText('The model returned no explanation.')).toBeInTheDocument()
    expect(screen.queryByText(/MALFORMED/)).not.toBeInTheDocument()
    expect(JSON.parse(localStorage.getItem('clearmed:v1:practice-answers')!)).toEqual({ first: 'B' })
  })

  it('reloads the prediction for a previously answered question', async () => {
    localStorage.setItem('clearmed:v1:practice-answers', JSON.stringify({ first: 'B' }))
    fetchMock.mockResolvedValueOnce(json([first])).mockResolvedValueOnce(prediction())
    show()
    expect(await screen.findByText('A model explanation.')).toBeInTheDocument()
    expect(screen.getByText('Correct.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Check answer' })).not.toBeInTheDocument()
  })

  it('ignores a late prediction after switching questions', async () => {
    let finish!: (response: Response) => void
    fetchMock.mockResolvedValueOnce(json([first, second]))
      .mockImplementationOnce(() => new Promise<Response>(resolve => { finish = resolve }))
    show()
    await answer()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await userEvent.click(screen.getByRole('button', { name: /Second question/ }))
    await act(async () => finish(prediction()))
    expect(screen.getByRole('heading', { name: 'Second question?' })).toBeInTheDocument()
    expect(screen.queryByText('A model explanation.')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Check answer' })).toBeDisabled()
  })
})
