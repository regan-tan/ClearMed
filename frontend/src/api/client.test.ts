import { afterEach, describe, expect, it, vi } from 'vitest'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.resetModules()
})

describe('backend connection', () => {
  it('calls the same-origin API', async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => [] })
    vi.stubGlobal('fetch', fetch)
    const { api } = await import('./client')
    await expect(api.listQuestions()).resolves.toEqual([])
    expect(fetch).toHaveBeenCalledWith('/api/questions', expect.any(Object))
  })

  it('reports an unavailable backend instead of showing substitute data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const { api } = await import('./client')
    await expect(api.listQuestions()).rejects.toThrow('Cannot reach the tutor backend')
  })
})

describe('backend response handling', () => {
  const question = { id: 'q1', question: 'Question?', options: { A: 'a', B: 'b', C: 'c', D: 'd' }, correct: 'B', subject: 'Test' }

  async function backend(response: Response) {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response))
    return (await import('./client')).api
  }

  it.each([
    {}, [question, question], [{ ...question, options: { A: 'a' } }], [{ ...question, correct: 'E' }],
  ])('rejects invalid question banks: %j', async body => {
    const api = await backend(new Response(JSON.stringify(body)))
    await expect(api.listQuestions()).rejects.toThrow('invalid question bank')
  })

  it.each([
    { question_id: 'other', predicted: 'B', explanation: 'Text' },
    { question_id: 'q1', predicted: 'MALFORMED', explanation: 'Text' },
  ])('rejects mismatched or malformed predictions: %j', async body => {
    const api = await backend(new Response(JSON.stringify(body)))
    await expect(api.predict('q1')).rejects.toThrow('invalid prediction')
  })

  it('encodes the question ID and accepts a prediction without an explanation', async () => {
    const api = await backend(new Response(JSON.stringify({ question_id: 'a/b', predicted: 'A', explanation: '' })))
    await expect(api.predict('a/b')).resolves.toMatchObject({ predicted: 'A', explanation: '' })
    expect(fetch).toHaveBeenCalledWith('/api/questions/a%2Fb/predict', expect.objectContaining({ method: 'POST' }))
  })

  it('handles a non-JSON proxy error', async () => {
    const api = await backend(new Response('Bad gateway', { status: 502, statusText: 'Bad Gateway' }))
    await expect(api.listQuestions()).rejects.toThrow('502: Bad Gateway')
  })

  it('rejects a non-JSON success response', async () => {
    const api = await backend(new Response('<html>Wrong server</html>'))
    await expect(api.listQuestions()).rejects.toThrow('invalid JSON response')
  })

  it.each([
    { reply: '', fallback: false, evidence: [] },
    { reply: 'Reply', fallback: 'false', evidence: [] },
    { reply: 'Reply', fallback: false, evidence: [{ id: 'd1', text: 'Text', subject: 'Test', score: '0.8' }] },
    { reply: 'Reply', fallback: false, evidence: [{ id: 'd1', text: 'Text', subject: 'Test', score: null }] },
    { reply: 'Reply', fallback: false, evidence: [1, 2] },
    { reply: 'Reply', fallback: false, evidence: Array(2).fill({ id: 'd1', text: 'Text', subject: 'Test', score: 0.8 }) },
  ])('rejects invalid follow-up responses: %j', async body => {
    const api = await backend(new Response(JSON.stringify(body)))
    await expect(api.followUp({ question_id: 'q1', history: [], message: 'Why?', embedding_model: 'minilm', top_k: 3 }))
      .rejects.toThrow('invalid follow-up response')
  })

  it('lists answer models and rejects an invalid or duplicated list', async () => {
    const models = [{ id: 'flan_t5_lora', label: 'FLAN-T5 + LoRA' }, { id: 'other', label: 'Other' }]
    expect(await (await backend(new Response(JSON.stringify(models)))).listModels()).toEqual(models)
    for (const bad of [{}, [{ id: 'a' }], [models[0], models[0]]]) {
      const api = await backend(new Response(JSON.stringify(bad)))
      await expect(api.listModels()).rejects.toThrow('invalid model list')
    }
  })

  it('sends the chosen model with predictions and omits it by default', async () => {
    const body = { question_id: 'q1', predicted: 'A', explanation: '' }
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => new Response(JSON.stringify(body))))
    const { api } = await import('./client')
    await api.predict('q1', 'other model')
    expect(fetch).toHaveBeenLastCalledWith('/api/questions/q1/predict?llm=other%20model', expect.anything())
    await api.predict('q1')
    expect(fetch).toHaveBeenLastCalledWith('/api/questions/q1/predict', expect.anything())
  })
})
