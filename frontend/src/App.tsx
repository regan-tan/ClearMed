import { useEffect, useState } from 'react'
import { api } from './api/client'
import type { EmbeddingModel, LlmModel } from './api/types'
import { AppHeader, type Mode } from './components/AppHeader'
import { FlashcardsView } from './components/FlashcardsView'
import { PracticeView } from './components/PracticeView'
import { createCard, type Flashcard, type FlashcardDraft } from './flashcards'
import { usePersistentState } from './storage'
import { useTheme } from './theme'

const MODES: Mode[] = ['practice', 'flashcards']

export default function App() {
  const { theme, toggle } = useTheme()
  const [storedMode, setMode] = usePersistentState<Mode>('mode', 'practice')
  // A mode saved by an older version (e.g. the removed Notebook mode) falls back to Practice.
  const mode = MODES.includes(storedMode) ? storedMode : 'practice'
  useEffect(() => {
    if (mode !== storedMode) setMode(mode)
  }, [mode, storedMode, setMode])
  const [model, setModel] = usePersistentState<EmbeddingModel>('model', 'e5')
  const [topK, setTopK] = usePersistentState('top-k', 3)
  // The server decides which answer models exist. If the list can't be loaded the picker stays hidden and
  // requests omit `llm`, so the server's default model answers.
  const [llmModels, setLlmModels] = useState<LlmModel[]>([])
  const [storedLlm, setLlm] = usePersistentState('llm', '')
  const llm = llmModels.find(option => option.id === storedLlm)?.id ?? llmModels[0]?.id
  useEffect(() => {
    let cancelled = false
    api
      .listModels()
      .then(list => {
        if (!cancelled) setLlmModels(list)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])
  const [cards, setCards] = usePersistentState<Flashcard[]>('flashcards', [])
  const cardOrigins = cards.flatMap(card => (card.origin ? [card.origin] : []))

  function addCard(draft: FlashcardDraft) {
    setCards(current =>
      draft.origin && current.some(card => card.origin === draft.origin) ? current : [createCard(draft), ...current],
    )
  }

  return (
    <div className="notebook-app">
      <AppHeader
        mode={mode}
        onModeChange={setMode}
        model={model}
        topK={topK}
        onModelChange={setModel}
        onTopKChange={setTopK}
        theme={theme}
        onToggleTheme={toggle}
        llmModels={llmModels}
        llm={llm}
        onLlmChange={setLlm}
      />

      {/* Kept mounted so MCQ progress survives switching modes */}
      <div className="mode-view" hidden={mode !== 'practice'}>
        <PracticeView model={model} topK={topK} llm={llm} flashcardOrigins={cardOrigins} onFlashcard={addCard} />
      </div>

      {mode === 'flashcards' && (
        <div className="mode-view">
          <FlashcardsView cards={cards} setCards={setCards} />
        </div>
      )}
    </div>
  )
}
