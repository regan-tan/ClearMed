import { describe, expect, it } from 'vitest'

import { FALLBACK_MESSAGE, mockApi } from './mock'

describe('mock tutor adapter', () => {
  it('aligns the Philadelphia chromosome prediction with its gold CML answer', async () => {
    const questions = await mockApi.listQuestions()
    const philadelphia = questions.find(question => question.id === 'q4')
    const prediction = await mockApi.predict('q4')

    expect(philadelphia?.correct).toBe('C')
    expect(prediction.predicted).toBe(philadelphia?.correct)
    expect(prediction.explanation).toMatch(/chronic myeloid leukaemia/i)
  })

  it('keeps retrieval previews and the insufficient-evidence fallback deterministic', async () => {
    const response = await mockApi.followUp({
      question_id: 'q1',
      history: [],
      message: 'unrelated words',
      embedding_model: 'e5',
      top_k: 2,
    })

    expect(response.fallback).toBe(true)
    expect(response.reply).toBe(FALLBACK_MESSAGE)
    expect(response.evidence).toHaveLength(2)
  })
})
