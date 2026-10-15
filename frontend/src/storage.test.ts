import { describe, expect, it } from 'vitest'

import { readStored, writeStored } from './storage'

describe('local storage persistence', () => {
  it('round-trips a notebook preview value under the established key prefix', () => {
    writeStored('notebook:preview:notes', [{ id: 'note-1', text: 'Keep this note' }])

    expect(readStored('notebook:preview:notes', [])).toEqual([{ id: 'note-1', text: 'Keep this note' }])
    expect(localStorage.getItem('clearmed:v1:notebook:preview:notes')).toContain('Keep this note')
  })
})
