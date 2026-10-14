import { ListChecks } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { EmbeddingModel, OptionKey, Question } from '../api/types'
import type { FlashcardDraft } from '../flashcards'
import { usePersistentState } from '../storage'
import { ChatPanel } from './ChatPanel'
import { McqCard } from './McqCard'
import { QuestionList } from './QuestionList'

interface Props {
  model: EmbeddingModel
  topK: number
  flashcardOrigins: string[]
  onFlashcard: (draft: FlashcardDraft) => void
}

// MCQ practice: question bank, then the MCQ card with its follow-up chat and retrieved evidence.
export function PracticeView({ model, topK, flashcardOrigins, onFlashcard }: Props) {
  const [questions, setQuestions] = useState<Question[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [reload, setReload] = useState(0)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [answers, setAnswers] = usePersistentState<Record<string, OptionKey>>('practice-answers', {})
  const [bankOpen, setBankOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    api
      .listQuestions()
      .then(qs => {
        if (!cancelled) {
          setQuestions(qs)
          setActiveId(id => id ?? qs[0]?.id ?? null)
        }
      })
      .catch(e => {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : 'Request failed')
      })
    return () => {
      cancelled = true
    }
  }, [reload])

  const active = questions.find(q => q.id === activeId)

  return (
    <>
      <div className="practice-tabs">
        <button onClick={() => setBankOpen(open => !open)} aria-expanded={bankOpen}>
          <ListChecks size={15} />
          Questions ({questions.length})
        </button>
      </div>
      <main className="practice-layout">
        <div className={`bank-slot ${bankOpen ? 'mobile-visible' : ''}`}>
          <QuestionList
            questions={questions}
            activeId={activeId}
            answers={answers}
            onSelect={id => {
              setActiveId(id)
              setBankOpen(false)
            }}
            onCollapse={() => setBankOpen(false)}
          />
        </div>
        {active ? (
          // key resets MCQ and chat state when switching questions
          <ChatPanel key={active.id} question={active} embeddingModel={model} topK={topK}>
            <McqCard
              question={active}
              answer={answers[active.id]}
              onAnswer={key => setAnswers(current => ({ ...current, [active.id]: key }))}
              inDeck={flashcardOrigins.includes(`mcq:${active.id}`)}
              onFlashcard={onFlashcard}
            />
          </ChatPanel>
        ) : (
          <section className="notebook-panel practice-status">
            {loadError ? (
              <>
                <p>Couldn’t load questions ({loadError}).</p>
                <button
                  className="text-link"
                  onClick={() => {
                    setLoadError(null)
                    setReload(r => r + 1)
                  }}
                >
                  Try again
                </button>
              </>
            ) : (
              <p>Loading questions…</p>
            )}
          </section>
        )}
      </main>
    </>
  )
}
