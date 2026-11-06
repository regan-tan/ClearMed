import { afterEach, describe, expect, it, vi } from 'vitest'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.resetModules()
})

describe('practice connection mode', () => {
  it.each([undefined, 'true'])('uses previews for %s', async setting => {
    vi.stubEnv('VITE_USE_MOCK', setting)
    const { api, practicePreview } = await import('./client')
    const { mockApi } = await import('./mock')
    expect(practicePreview).toBe(true)
    expect(api).toBe(mockApi)
  })

  it('uses the same-origin API when backend mode is enabled', async () => {
    vi.stubEnv('VITE_USE_MOCK', 'false')
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => [] })
    vi.stubGlobal('fetch', fetch)
    const { api, practicePreview } = await import('./client')
    expect(practicePreview).toBe(false)
    await expect(api.listQuestions()).resolves.toEqual([])
    expect(fetch).toHaveBeenCalledWith('/api/questions', expect.any(Object))
  })

  it('does not substitute mock data when the backend is unavailable', async () => {
    vi.stubEnv('VITE_USE_MOCK', 'false')
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const { api } = await import('./client')
    await expect(api.listQuestions()).rejects.toThrow('Failed to fetch')
  })

  it('rejects a mistyped mode instead of silently showing mock data', async () => {
    vi.stubEnv('VITE_USE_MOCK', 'False')
    await expect(import('./client')).rejects.toThrow('VITE_USE_MOCK must be')
  })
})
