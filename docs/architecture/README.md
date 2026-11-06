# Architecture

ClearMed is a Python 3.11 `src/clearmed` package with thin CLI scripts in `scripts/`, plus an
independent React + Vite frontend in `frontend/`. Part I and Part II share the data layer
(`clearmed.data`) and the seed-42 partitions. Part II reuses Part I's frozen FLAN-T5 + LoRA
checkpoint but does not retrain it.

## Part I: answer and explain an MCQ

```text
MedMCQA official train (182,822)
  -> seed-42 partitions: part1_train / internal_dev / retrieval_kb
  -> part1_train trains TF-IDF + LR, scratch Seq2Seq, FLAN-T5 + LoRA
  -> internal_dev selects checkpoints
  -> frozen models scored once on official validation (4,183)
  -> RQ1 accuracy (clearmed.evaluation.qa), RQ2 ROUGE-L / BERTScore
```

## Part II: answer a follow-up from retrieved evidence

```text
retrieval_kb (36,564) minus empty explanations -> 32,218 passages
  -> embedded once per model: MiniLM / E5-base-v2 / BioLORD-2023

student MCQ + follow-up
  1. Part I FLAN-T5 + LoRA predicts A-D (predicted answer, never the gold one)
  2. query = follow-up + predicted answer + question and options
  3. cosine search, keep top 5
  4. top-1 score >= model threshold?
       no  -> fixed fallback message (not generated)
       yes -> 5. plain FLAN-T5-base (adapter off) reads top 3 passages -> "Answer: X ..." reply
```

RQ3 scores step 3 on its own, with the gold answer in the query so that every model gets the
same correct context. RQ4 scores the whole pipeline with the predicted answer, against a
no-retrieval control.

## Code map

| Step | Module |
|---|---|
| Partitions, leakage checks | `clearmed.data.splitting`, `clearmed.data.verification` |
| MCQ formatting | `clearmed.data.preprocessing` |
| Part I models | `clearmed.baselines.tfidf`, `clearmed.models`, `clearmed.training` |
| Passage bank, embedders, index | `clearmed.retrieval.corpus`, `.embedders`, `.index` |
| Query building, confidence check | `clearmed.retrieval.queries`, `.gate` |
| Tutor pipeline, RQ4 rubric | `clearmed.conversation.pipeline`, `.rubric` |
| Metrics | `clearmed.evaluation.qa`, `.retrieval`, `.conversation` |
| Demo backend | `clearmed.app.demo` (served by `scripts/serve_demo.py`) |
