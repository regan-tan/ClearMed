import clsx from 'clsx'
import { Check, Layers, Loader2, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { McqPrediction, OptionKey, Question } from '../api/types'
import type { FlashcardDraft } from '../flashcards'

const KEYS: OptionKey[] = ['A', 'B', 'C', 'D']

interface Props {
  question: Question
  answer?: OptionKey
  onAnswer: (key: OptionKey) => void
  inDeck: boolean
  onFlashcard: (draft: FlashcardDraft) => void
}

export function McqCard({ question, answer, onAnswer, inDeck, onFlashcard }: Props) {
  const [selected, setSelected] = useState<OptionKey | null>(answer ?? null)
  const [prediction, setPrediction] = useState<McqPrediction | null>(null)
  const [error, setError] = useState<string | null>(null)

  const answered = answer !== undefined

  const [retry, setRetry] = useState(0)
  useEffect(() => {
    if (!answered) return
    let cancelled = false
    api
      .predict(question.id)
      .then(result => {
        if (!cancelled) setPrediction(result)
      })
      .catch(e => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Request failed')
      })
    return () => {
      cancelled = true
    }
  }, [answered, question.id, retry])

  return (
    <section>
      <div className="flex items-center gap-2 text-xs text-fg-muted">
        <span className="font-semibold text-accent">{question.subject}</span>
        {question.topic && <span>{question.topic}</span>}
      </div>
      <h2 className="question-heading mt-4">{question.question}</h2>
      <p className="mt-4 text-xs text-fg-muted">Select the single best answer.</p>

      <div className="mt-5 grid gap-2">
        {KEYS.map(key => {
          const isCorrect = key === question.correct
          const isPicked = key === selected
          return (
            <button
              key={key}
              onClick={() => setSelected(key)}
              aria-pressed={selected === key}
              disabled={answered}
              className={clsx(
                'flex items-center gap-3 rounded-md border px-4 py-3 text-left text-sm transition',
                !answered &&
                  (isPicked
                    ? 'border-primary bg-accent-soft ring-1 ring-primary'
                    : 'border-border hover:border-primary hover:bg-subtle'),
                // Answered options are disabled; keep the correct and picked ones at full strength.
                answered && (isCorrect || isPicked) && 'opacity-100',
                answered && isCorrect && 'border-primary bg-accent-soft',
                answered &&
                  isPicked &&
                  !isCorrect &&
                  'border-rose-400 bg-rose-50 dark:border-rose-500/50 dark:bg-rose-500/10',
                answered && !isCorrect && !isPicked && 'border-border opacity-60',
              )}
            >
              <span
                className={clsx(
                  'flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-semibold',
                  answered && isCorrect
                    ? 'bg-primary text-primary-fg'
                    : answered && isPicked
                      ? 'bg-rose-500 text-white'
                      : 'bg-subtle text-fg-muted',
                )}
              >
                {answered && isCorrect ? (
                  <Check className="h-4 w-4" />
                ) : answered && isPicked ? (
                  <X className="h-4 w-4" />
                ) : (
                  key
                )}
              </span>
              <span>{question.options[key]}</span>
            </button>
          )
        })}
      </div>

      {!answered && (
        <div className="mt-5 flex items-center gap-4">
          <button
            className="primary-button"
            disabled={!selected}
            onClick={() => {
              if (selected) onAnswer(selected)
            }}
          >
            Check answer
          </button>
          <span className="text-xs text-fg-muted">
            {selected ? `Option ${selected} selected` : 'Choose an option to continue'}
          </span>
        </div>
      )}

      {answered && (
        <div className="mt-5 rounded-md bg-subtle p-5" aria-live="polite">
          <p
            className={clsx(
              'text-sm font-semibold',
              selected === question.correct ? 'text-accent' : 'text-rose-700 dark:text-rose-400',
            )}
          >
            {selected === question.correct ? 'Correct.' : `Not quite. The answer is ${question.correct}.`}
          </p>

          <div className="mt-3 border-t border-border pt-3">
            <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-fg-muted">
              Answer explanation
            </div>
            {error ? (
              <div>
                <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">
                  Couldn't load the explanation ({error}).
                </p>
                <button
                  className="mt-2 text-sm underline"
                  onClick={() => {
                    setError(null)
                    setRetry(r => r + 1)
                  }}
                >
                  Try again
                </button>
              </div>
            ) : !prediction ? (
              <p className="mt-2 flex items-center gap-2 text-sm text-fg-muted">
                <Loader2 className="h-4 w-4 animate-spin" /> Generating…
              </p>
            ) : (
              <>
                <p className="mt-2 text-sm">
                  Model predicted <strong>{prediction.predicted}</strong>{' '}
                  {prediction.predicted === question.correct ? (
                    <span className="text-accent">(matches the answer key)</span>
                  ) : (
                    <span className="text-rose-600 dark:text-rose-400">(differs from the answer key)</span>
                  )}
                </p>
                <p className="mt-1.5 text-sm leading-relaxed text-fg">{prediction.explanation}</p>
              </>
            )}
          </div>
          <button
            className="mt-4 flex items-center gap-1.5 text-xs font-medium text-fg-muted transition hover:text-accent"
            disabled={inDeck}
            onClick={() =>
              onFlashcard({
                front: question.question,
                back: `${question.correct}. ${question.options[question.correct]}${prediction ? `\n\n${prediction.explanation}` : ''}`,
                label: question.subject,
                origin: `mcq:${question.id}`,
              })
            }
          >
            <Layers className="h-3.5 w-3.5" /> {inDeck ? 'In flashcards' : 'Add to flashcards'}
          </button>
        </div>
      )}
    </section>
  )
}
