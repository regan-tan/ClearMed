import clsx from 'clsx'
import { Check, Loader2, Sparkles, X } from 'lucide-react'
import { useState } from 'react'
import { api } from '../api/client'
import type { McqPrediction, OptionKey, Question } from '../api/types'

const KEYS: OptionKey[] = ['A', 'B', 'C', 'D']

export function McqCard({ question }: { question: Question }) {
  const [selected, setSelected] = useState<OptionKey | null>(null)
  const [prediction, setPrediction] = useState<McqPrediction | null>(null)
  const [error, setError] = useState<string | null>(null)

  const answered = selected !== null

  async function choose(key: OptionKey) {
    if (answered) return
    setSelected(key)
    try {
      setPrediction(await api.predict(question.id))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    }
  }

  return (
    <section className="rounded-2xl bg-white p-6 shadow-sm">
      <div className="flex items-center gap-2 text-xs text-slate-500">
        <span className="rounded-full bg-brand-50 px-2 py-0.5 font-medium text-brand-700">
          {question.subject}
        </span>
        {question.topic && <span>{question.topic}</span>}
      </div>
      <h1 className="mt-3 text-lg font-semibold leading-snug">{question.question}</h1>

      <div className="mt-5 grid gap-2">
        {KEYS.map((key) => {
          const isCorrect = key === question.correct
          const isPicked = key === selected
          return (
            <button
              key={key}
              onClick={() => choose(key)}
              disabled={answered}
              className={clsx(
                'flex items-center gap-3 rounded-xl border px-4 py-3 text-left text-sm transition',
                !answered && 'border-slate-200 hover:border-brand-500 hover:bg-brand-50',
                answered && isCorrect && 'border-emerald-400 bg-emerald-50',
                answered && isPicked && !isCorrect && 'border-rose-400 bg-rose-50',
                answered && !isCorrect && !isPicked && 'border-slate-200 opacity-60',
              )}
            >
              <span
                className={clsx(
                  'flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-semibold',
                  answered && isCorrect
                    ? 'bg-emerald-500 text-white'
                    : answered && isPicked
                      ? 'bg-rose-500 text-white'
                      : 'bg-slate-100 text-slate-600',
                )}
              >
                {answered && isCorrect ? <Check className="h-4 w-4" /> : answered && isPicked ? <X className="h-4 w-4" /> : key}
              </span>
              <span>{question.options[key]}</span>
            </button>
          )
        })}
      </div>

      {answered && (
        <div className="mt-5 rounded-xl bg-slate-50 p-4">
          <p className={clsx('text-sm font-semibold', selected === question.correct ? 'text-emerald-700' : 'text-rose-700')}>
            {selected === question.correct ? 'Correct!' : `Not quite. The answer is ${question.correct}.`}
          </p>

          <div className="mt-3 border-t border-slate-200 pt-3">
            <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-500">
              <Sparkles className="h-3.5 w-3.5 text-brand-500" /> ClearMed explanation
            </div>
            {error ? (
              <p className="mt-2 text-sm text-rose-600">Couldn't load the model's explanation ({error}).</p>
            ) : !prediction ? (
              <p className="mt-2 flex items-center gap-2 text-sm text-slate-500">
                <Loader2 className="h-4 w-4 animate-spin" /> Generating…
              </p>
            ) : (
              <>
                <p className="mt-2 text-sm">
                  Model predicted <strong>{prediction.predicted}</strong>{' '}
                  {prediction.predicted === question.correct ? (
                    <span className="text-emerald-600">(matches the answer key)</span>
                  ) : (
                    <span className="text-rose-600">(differs from the answer key)</span>
                  )}
                </p>
                <p className="mt-1.5 text-sm leading-relaxed text-slate-700">{prediction.explanation}</p>
              </>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
