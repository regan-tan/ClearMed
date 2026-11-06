import { act, cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ChatPanel } from './ChatPanel'

vi.hoisted(() => vi.stubEnv('VITE_USE_MOCK', 'false'))

const question = {
  id: 'q1', question: 'Which option?', subject: 'Test', correct: 'B' as const,
  options: { A: 'Alpha', B: 'Beta', C: 'Gamma', D: 'Delta' },
}
const evidence = [{ id: 'doc-7', text: 'Returned passage text.', subject: 'Test', score: 0.87654 }]
const fetchMock = vi.fn()
const response = (reply = 'Tutor reply.', fallback = false, passages = evidence) =>
  new Response(JSON.stringify({ reply, fallback, evidence: passages }))
const requestBody = (index: number) => JSON.parse(fetchMock.mock.calls[index][1].body)

async function send(text: string) {
  await userEvent.type(screen.getByRole('textbox', { name: 'Follow-up question' }), text)
  await userEvent.click(screen.getByRole('button', { name: 'Send' }))
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('follow-up HTTP integration', () => {
  it('sends only conversation context and selected settings, and renders returned evidence', async () => {
    fetchMock.mockResolvedValueOnce(response()).mockResolvedValueOnce(response('Second reply.'))
    const view = render(<ChatPanel question={question} embeddingModel="e5" topK={2} />)
    await send('Why?')
    expect(await screen.findByText('Tutor reply.')).toBeInTheDocument()
    expect(fetchMock.mock.calls[0][0]).toBe('/api/follow-up')
    expect(requestBody(0)).toEqual({ question_id: 'q1', history: [], message: 'Why?', embedding_model: 'e5', top_k: 2 })
    await userEvent.click(screen.getByRole('button', { name: '1 retrieved passage' }))
    expect(screen.getByText('Returned passage text.')).toBeInTheDocument()
    expect(screen.getByText('doc-7')).toBeInTheDocument()
    expect(screen.getByText('score 0.877')).toBeInTheDocument()
    view.rerender(<ChatPanel question={question} embeddingModel="biolord" topK={5} />)
    await send('Compare them.')
    await screen.findByText('Second reply.')
    expect(requestBody(1)).toEqual({
      question_id: 'q1', message: 'Compare them.', embedding_model: 'biolord', top_k: 5,
      history: [{ role: 'user', content: 'Why?' }, { role: 'assistant', content: 'Tutor reply.' }],
    })
  })

  it('retries the exact failed request without duplicating turns or sending errors as history', async () => {
    fetchMock.mockResolvedValueOnce(response('Earlier reply.'))
      .mockRejectedValueOnce(new TypeError('Network failure'))
      .mockResolvedValueOnce(response('Retried reply.'))
      .mockResolvedValueOnce(response('Final reply.'))
    const view = render(<ChatPanel question={question} embeddingModel="minilm" topK={3} />)
    await send('First turn')
    await screen.findByText('Earlier reply.')
    await send('Second turn')
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the tutor backend')
    expect(screen.getByText('Earlier reply.')).toBeInTheDocument()
    view.rerender(<ChatPanel question={question} embeddingModel="e5" topK={1} />)
    await userEvent.click(screen.getByRole('button', { name: 'Retry follow-up' }))
    await screen.findByText('Retried reply.')
    expect(requestBody(2)).toEqual(requestBody(1))
    expect(screen.getAllByText('Second turn')).toHaveLength(1)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    await send('Third turn')
    await screen.findByText('Final reply.')
    expect(requestBody(3).history).toEqual([
      { role: 'user', content: 'First turn' }, { role: 'assistant', content: 'Earlier reply.' },
      { role: 'user', content: 'Second turn' }, { role: 'assistant', content: 'Retried reply.' },
    ])
    expect(requestBody(3).embedding_model).toBe('e5')
  })

  it.each([true, false])('shows a fallback with passages=%s as insufficient evidence, not an error', async hasPassages => {
    fetchMock.mockResolvedValueOnce(response('I could not find reliable evidence.', true, hasPassages ? evidence : []))
    render(<ChatPanel question={question} embeddingModel="minilm" topK={3} />)
    await send('Explain')
    expect(await screen.findByText('Insufficient evidence')).toBeInTheDocument()
    expect(screen.getByText('I could not find reliable evidence.')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByText(hasPassages
      ? 'These passages did not meet the evidence confidence threshold.'
      : 'No passages were returned.')).toBeInTheDocument()
  })

  it('blocks duplicate sends while a reply is pending', async () => {
    let finish!: (res: Response) => void
    fetchMock.mockImplementationOnce(() => new Promise<Response>(resolve => { finish = resolve }))
    render(<ChatPanel question={question} embeddingModel="minilm" topK={3} />)
    await userEvent.dblClick(screen.getByRole('button', { name: 'Explain the reasoning' }))
    expect(screen.getByRole('status')).toHaveTextContent('Preparing tutor reply')
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    await act(async () => finish(response()))
    await screen.findByText('Tutor reply.')
  })

  it('does not show a late reply in another question', async () => {
    let finish!: (res: Response) => void
    fetchMock.mockImplementationOnce(() => new Promise<Response>(resolve => { finish = resolve }))
    const view = render(<ChatPanel key="q1" question={question} embeddingModel="minilm" topK={3} />)
    await send('Old question')
    view.rerender(<ChatPanel key="q2" question={{ ...question, id: 'q2' }} embeddingModel="minilm" topK={3} />)
    await act(async () => finish(response('Old reply.')))
    expect(screen.queryByText('Old reply.')).not.toBeInTheDocument()
    expect(screen.queryByText('Old question')).not.toBeInTheDocument()
  })

  it('displays invalid response failures separately from tutor messages', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ reply: 'Unvalidated reply', fallback: false, evidence: [{}] })))
    render(<ChatPanel question={question} embeddingModel="minilm" topK={3} />)
    await send('Explain')
    expect(await screen.findByRole('alert')).toHaveTextContent('invalid follow-up response')
    expect(screen.queryByText('Unvalidated reply')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry follow-up' })).toBeInTheDocument()
  })
})
