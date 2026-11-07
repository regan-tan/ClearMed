import type { FollowUpResponse, LlmModel, McqPrediction, Question, TutorApi } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    throw new Error('Cannot reach the tutor backend. Check that it is running and try again.')
  }
  if (!res.ok) {
    const body: unknown = await res.json().catch(() => null)
    const detail = isRecord(body) && typeof body.detail === 'string' ? body.detail : res.statusText
    throw new Error(`${res.status}${detail ? `: ${detail}` : ''}`)
  }
  try {
    return await res.json() as T
  } catch {
    throw new Error('The tutor backend returned an invalid JSON response.')
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isLabel(value: unknown) {
  return typeof value === 'string' && ['A', 'B', 'C', 'D'].includes(value)
}

function isText(value: unknown) {
  return typeof value === 'string' && value.trim().length > 0
}

function hasFourOptions(value: unknown) {
  return isRecord(value) && ['A', 'B', 'C', 'D'].every(key => isText(value[key]))
}

export const api: TutorApi = {
  listQuestions: async () => {
    const data = await request<unknown>('/questions')
    if (!Array.isArray(data) || data.some(q =>
      !isRecord(q) || !isText(q.id) || !isText(q.question) || !isLabel(q.correct) ||
      typeof q.subject !== 'string' || (q.topic !== undefined && q.topic !== null && typeof q.topic !== 'string') ||
      !hasFourOptions(q.options)
    ) || new Set(data.map(q => q.id)).size !== data.length) {
      throw new Error('The tutor backend returned an invalid question bank.')
    }
    return data as Question[]
  },
  listModels: async () => {
    const data = await request<unknown>('/models')
    if (!Array.isArray(data) || data.some(m => !isRecord(m) || !isText(m.id) || !isText(m.label)) ||
      new Set(data.map(m => m.id)).size !== data.length) {
      throw new Error('The tutor backend returned an invalid model list.')
    }
    return data as LlmModel[]
  },
  predict: async (questionId, llm) => {
    const query = llm ? `?llm=${encodeURIComponent(llm)}` : ''
    const data = await request<unknown>(`/questions/${encodeURIComponent(questionId)}/predict${query}`, { method: 'POST' })
    if (!isRecord(data) || data.question_id !== questionId || !isLabel(data.predicted) || typeof data.explanation !== 'string') {
      throw new Error('The tutor backend returned an invalid prediction for this question.')
    }
    return data as unknown as McqPrediction
  },
  followUp: async req => {
    const data = await request<unknown>('/follow-up', { method: 'POST', body: JSON.stringify(req) })
    if (!isRecord(data) || !isText(data.reply) || typeof data.fallback !== 'boolean' ||
      !Array.isArray(data.evidence) || data.evidence.some(e =>
        !isRecord(e) || !isText(e.id) || !isText(e.text) || typeof e.subject !== 'string' ||
        typeof e.score !== 'number' || !Number.isFinite(e.score)
      ) || new Set(data.evidence.map(e => e.id)).size !== data.evidence.length) {
      throw new Error('The tutor backend returned an invalid follow-up response.')
    }
    return data as unknown as FollowUpResponse
  },
}
