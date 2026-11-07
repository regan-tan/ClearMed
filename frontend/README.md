# ClearMed frontend

Practice MCQs against the ClearMed backend and revise them with flashcards.
React + TypeScript + Vite + Tailwind CSS.

## Run

The frontend always calls the real backend; there is no mock mode. Start the backend
from the repository root (it needs the existing checkpoint, corpus, retrieval indexes
and cached dataset; see [the demo guide](../docs/demo.md)):

```bash
uv run python scripts/serve_demo.py --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best
```

Then, in this directory:

```bash
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. To use another backend origin, copy
`.env.example` to `.env.local` and set `CLEARMED_BACKEND_URL` (origin only, no `/api`),
then restart Vite. `.env.local` is ignored by Git. Production hosting must route
same-origin `/api` requests to FastAPI; the proxy is for the development server only.

If the backend is not running, Practice shows "Cannot reach the tutor backend" with a
retry button. Nothing is substituted.

## Modes

The header switches between **Practice MCQ** and **Flashcards**, and has the search
settings (search model, **Passages shown**, and an answer model when the backend offers
more than one) and the light/dark theme. The search model defaults to MiniLM, which won RQ3.

- **Practice MCQ:** question bank, the MCQ card (select an option, then _Check answer_
  to reveal the answer key and the model explanation) and a follow-up chat with
  retrieved evidence. The chat input stays anchored at the bottom while the conversation
  scrolls. Drag the divider beside the question list to resize it (arrow keys also work,
  double-click resets; the width is saved and shared with the Flashcards deck). Answers are
  saved in this browser. An answered question can be added to flashcards.
- **Flashcards:** cards come from practice questions. Due cards are shown one at a time
  as a flip card. Click or press <kbd>Space</kbd> to flip, then grade with _Again_
  (<kbd>1</kbd>, back to box 0 and repeated this session) or _Got it_ (<kbd>2</kbd>,
  next review in 1, 3, 7, 14, then 30 days). _Study all_ goes through every card without
  rescheduling. Cards can be searched and deleted with undo. Animations are skipped when
  the system asks for reduced motion.

Flashcards are stored in localStorage under `clearmed:v1:flashcards`. A deck saved by an
older version (per notebook) is copied into it once on load; the old keys are left alone.
A saved mode from the removed Notebook view falls back to Practice.

## API

The adapter is [`src/api/client.ts`](src/api/client.ts); types are in
[`src/api/types.ts`](src/api/types.ts) and must match the FastAPI models in
`src/clearmed/app/demo.py`.

| Method | Path                          | Body              | Returns            |
| ------ | ----------------------------- | ----------------- | ------------------ |
| GET    | `/api/questions`              | –                 | `Question[]`       |
| GET    | `/api/models`                 | –                 | `LlmModel[]`       |
| POST   | `/api/questions/{id}/predict` | – (`?llm=<id>`)   | `McqPrediction`    |
| POST   | `/api/follow-up`              | `FollowUpRequest` | `FollowUpResponse` |

`/api/models` lists the answer/reply models the server can use; the first is the default.
The header's search settings show an **Answer model** picker only when there is more than one.
The chosen id is sent as `llm` (query parameter on predict, field on follow-up); leaving it out
uses the default, and an unknown id is a 404, never a silent fallback. The server registers
extra models with `TutorService.add_model` (see `src/clearmed/app/demo.py`).

`FollowUpResponse.fallback` is `true` when retrieval found insufficient evidence; `reply`
is then the fixed fallback message, not model output. `evidence` lists the passages the
reply was written from, capped at `top_k`. The **Passages shown** setting only changes how
many are displayed; the reply is always written from the same top passages. The MedGemma demo
server (see below) shows the top-k retrieved hits instead.

Responses are validated before rendering. Empty banks and request failures offer
retries. Prediction failure preserves the student's saved answer; follow-up retries
preserve the original message, history and search settings. Request errors never become
assistant turns in later history. For the startup, health and proxy checks, see
[the integration guide](../docs/demo.md#check-the-complete-connection).

## Using the MedGemma 27B demo server

The group's MedGemma 27B + MiniLM server runs on a GPU node and exposes the same routes, so the
frontend works with it unchanged. Its code is provided separately and is not in this repository
(`ClearMed_MedGemma27B_MiniLM_Demo.zip`). Differences from `serve_demo.py`: MiniLM only (E5 and
BioLORD return a 422 error), the confidence gate is off, and there is no `/api/models`, so the
answer-model picker stays hidden. Setup, the SSH tunnel and a checklist are in
[the demo guide](../docs/demo.md#alternative-medgemma-27b-server).

## Project structure

```
src/
  App.tsx            Mode switch, settings and the flashcard deck
  api/               client.ts (HTTP adapter) and types.ts (contract)
  hooks/useDismiss.ts  Close popovers on Escape or outside click
  components/        AppHeader, PracticeView, McqCard, ChatPanel, QuestionList,
                     EvidenceList, FlashcardsView, SearchSettings, ColumnResizer, Markdown
  bankWidth.ts       Limits for the resizable left panel
  flashcards.ts      Flashcard type, scheduling and legacy-deck migration
  storage.ts         localStorage helpers and usePersistentState
  theme.ts, utils.ts Light/dark theme; plural helper
  index.css          Design tokens and component styles
```

## Checks

```bash
npm run test          # Vitest
npm run lint          # oxlint
npm run build         # type-check and production build
npm run format:check  # Prettier (npm run format to fix)
```
