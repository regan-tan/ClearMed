import { mockApi } from './mock'
import type { TutorApi } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

const httpApi: TutorApi = {
  listQuestions: () => request('/questions'),
  predict: (questionId) => request(`/questions/${questionId}/predict`, { method: 'POST' }),
  followUp: (req) => request('/follow-up', { method: 'POST', body: JSON.stringify(req) }),
}

// Set VITE_USE_MOCK=false once the FastAPI backend is running.
export const api: TutorApi = import.meta.env.VITE_USE_MOCK === 'false' ? httpApi : mockApi
