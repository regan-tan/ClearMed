# ClearMed frontend

A resource-first notebook: add study material, select sources, chat, keep notes, practise MCQs and revise with flashcards.
React + TypeScript + Vite + Tailwind CSS. All notebook behavior currently uses a UI-only adapter.

## Run

```bash
cd frontend
npm install
npm run dev
```

## Current preview

- Multiple notebooks: each has its own sources, chats, notes and flashcards. Switch, create, rename (edit the title in place) and delete from the menu next to the title. Practice MCQ progress, search settings and theme are shared.
- Light/dark themes (follows the system setting until you pick one).
- Notebook, Practice MCQ and Flashcards modes.
- Multiple-file picker and drag/drop for PDF, DOCX, TXT and Markdown.
- File type/size checks (non-empty, up to 20 MB), source selection, and deletion with confirmation (optionally removing flashcards made from the source).
- Central chat with multiline input; works with or without selected sources (Enter sends, Shift+Enter adds a line).
- VS Code-style left sidebar with collapsible **Sources** and **Chats** sections and a draggable divider between them (arrow keys also resize; double-click resets). Collapsed state and split are remembered.
- Resizable columns: drag the dividers between the sidebar, chat and notes panels (arrow keys also resize; double-click resets to 280px). Widths are remembered, and the chat column always keeps at least 360px.
- Chat history: conversations are saved with a title; reopen, delete or start a new chat from the Chats section (or the history button on phones).
- Save responses to notes or write your own notes.
- Flashcards generated from selected sources, reviewed with a simple spaced-repetition schedule (Again / Got it). Ask follow-up questions about the card on screen. Chat answers and practice MCQs can also be added as cards in one click; cards can be deleted with undo.
- Responsive layout: side panels become overlays on narrow screens.
- Citation buttons and a passage viewer ready to render backend citations.

**Files are only staged as metadata.** The preview does not read, upload, parse, embed or search file contents. Chat returns a clearly labelled interface preview with no fabricated citations. Generated flashcards are clearly labelled placeholders. Notebooks and their data (source metadata, chats, notes, flashcards), plus practice answers, are saved in this browser's localStorage via `src/storage.ts` and `src/notebooks.ts`. Data saved before notebooks existed is moved into a first notebook on load. It survives refreshes but not clearing site data or switching devices. File size/type checks are convenience checks; the backend must validate uploads independently.

## Backend integration

Start in [`src/api/notebook.ts`](src/api/notebook.ts). It defines `NotebookApi`, source status, chat request/response and citation types. Replace `notebookApi` with an HTTP implementation; UI components do not need endpoint details. This adapter is separate from the older MCQ adapter; `VITE_USE_MOCK` currently affects only that older adapter, not the notebook.

| Adapter method                    | Proposed endpoint (not implemented)           | Contract                                                                                                                                 |
| --------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `listNotebooks()`                 | `GET /api/notebooks`                          | Return `NotebookSummary[]` (`id`, `title`, `createdAt`)                                                                                  |
| `createNotebook(title)`           | `POST /api/notebooks`                         | Return the new `NotebookSummary`; the app uses its `id`                                                                                  |
| `renameNotebook(id, title)`       | `PATCH /api/notebooks/:id`                    | Resolve after saving the title                                                                                                           |
| `deleteNotebook(id)`              | `DELETE /api/notebooks/:id`                   | Delete the notebook and everything in it                                                                                                 |
| `uploadSource(id, file)`          | `POST /api/notebooks/:id/sources`             | Send `File` as multipart FormData; return `NotebookSource`                                                                               |
| `listSources(id)`                 | `GET /api/notebooks/:id/sources`              | Return `NotebookSource[]`                                                                                                                |
| `removeSource(id, sourceId)`      | `DELETE /api/notebooks/:id/sources/:sourceId` | Resolve after successful deletion                                                                                                        |
| `chat(id, request)`               | `POST /api/notebooks/:id/chat`                | Send message, selected source IDs, prior history and optional `context` (e.g. the flashcard being revised); return content and citations |
| `generateFlashcards(id, request)` | `POST /api/notebooks/:id/flashcards`          | Send source IDs; return `{ front, back, source_id, source_name }[]` written from those sources                                           |

These are proposed contracts, not claims about existing backend routes. Add notebook identity/authentication to your adapter when implementing persistence. Do not set a JSON Content-Type header for FormData uploads; the browser supplies the multipart boundary.

### Source lifecycle

`status` supports `staged`, `processing`, `ready`, and `error`; `error` optionally contains a useful message. Only `ready` sources (and `staged` ones in the preview) can be ticked or sent with a request; processing sources show a spinner and failed ones show their error with a prompt to remove and re-upload. With a real adapter, `useNotebook` loads the source list with `listSources()` whenever a notebook opens (with loading and retry states), polls it every 3 seconds while any source is `processing`, and ticks sources as they become ready.

### Chat contract

```ts
{
  message: 'Explain this concept',
  source_ids: ['server-source-id'],
  history: [{ role: 'user', content: 'Earlier question' }],
  retrieval: { embedding_model: 'e5', top_k: 3 }
}
```

The current message is separate from history. `retrieval` carries the search settings from the header (also used for MCQ follow-ups). `source_ids` may be empty: chat works without uploaded sources, and the backend should then answer without user material (no source citations). The server should validate source ownership and perform parsing, retrieval, ranking, generation and grounding. None of that belongs in this UI adapter.

Return:

```ts
{
  content: 'Your grounded explanation…',
  citations: [{
    source_id: 'server-source-id',
    source_name: 'Lecture notes.pdf',
    excerpt: 'Exact supporting passage returned by the backend.',
    page: 12 // optional
  }]
}
```

Citation buttons open the passage in the right panel. Assistant replies, MCQ explanations, notes and flashcard backs render as Markdown (including tables) via `react-markdown`; raw HTML in replies is not rendered. Errors are displayed without dropping earlier conversation; source selection is captured when the request is sent.

### Before enabling real service behavior

- Replace the adapter and set its `preview` flag to `false`. That hides the preview labels, loads notebooks with `listNotebooks()` (creating a first one if the list is empty) and sources with `listSources()`, each with loading and retry states, and turns on source status polling.
- Chats, notes and flashcards (with review schedule) still live in localStorage; the contract has no endpoints for them yet. Per-notebook state and actions live in `src/hooks/useNotebook.ts`, the single place to swap out.
- Add request cancellation/streaming if supported by your backend.

## Practice MCQ mode

The header switch toggles between **Notebook**, **Practice MCQ** and **Flashcards**. Practice mode shows the question bank, the MCQ card (select an option, then _Check answer_ to reveal the answer key and the model explanation) and a follow-up chat with retrieved evidence. The sliders button in the header opens the search settings (embedding model and top-k). They apply in every mode: notebook chat, flashcard questions and MCQ follow-ups. Answers are saved locally, and an answered question can be added to flashcards.

Practice mode uses the MCQ adapter in [`src/api/client.ts`](src/api/client.ts) (types in [`src/api/types.ts`](src/api/types.ts)), which defaults to the mock in [`src/api/mock.ts`](src/api/mock.ts). Run with `VITE_USE_MOCK=false` to call the FastAPI backend; Vite proxies `/api/*` to `http://localhost:8000`.

| Method | Path                          | Body              | Returns            |
| ------ | ----------------------------- | ----------------- | ------------------ |
| GET    | `/api/questions`              | –                 | `Question[]`       |
| POST   | `/api/questions/{id}/predict` | –                 | `McqPrediction`    |
| POST   | `/api/follow-up`              | `FollowUpRequest` | `FollowUpResponse` |

`FollowUpResponse.fallback` is `true` when retrieval found insufficient evidence; `reply` is then the fixed fallback message, not model output.

## Flashcards mode

Cards live in `src/flashcards.ts` (type, Leitner-style scheduling) and are rendered by [`src/components/FlashcardsView.tsx`](src/components/FlashcardsView.tsx).

- **Generate:** tick sources in the deck panel and press _Generate flashcards_, which calls `notebookApi.generateFlashcards`. The preview adapter returns labelled placeholder cards.
- **Review:** due cards are shown one at a time as a 3D flip card, with the rest of the session stacked behind it. Click or press <kbd>Space</kbd> to flip between question and answer, then grade with _Again_ (<kbd>1</kbd>, back to box 0 and repeated this session) or _Got it_ (<kbd>2</kbd>, next review in 1, 3, 7, 14, then 30 days); the card slides out left or right and the next one deals in. Grade buttons stay disabled until the answer is shown. _Study all_ goes through every card without rescheduling. Animations are skipped when the system asks for reduced motion.
- **Ask about a card:** after flipping, the card chat (`FlashcardChat.tsx`) sends the card as `context` to `notebookApi.chat`, grounded in the card's source when it has one.

## Project structure

```
src/
  App.tsx              Page layout, notebook management (switch/create/rename/delete) and dialogs
  api/                 Backend adapters: notebook.ts (notebook contract + preview adapter),
                       client.ts / mock.ts / types.ts (MCQ practice)
  hooks/
    useNotebook.ts     One notebook's sources, chats, notes, flashcards and their actions
    useDismiss.ts      Close popovers on Escape or outside click
  components/          UI: AppHeader, NotebookSidebar, NotebookChat, NotesPanel, FlashcardsView, PracticeView, …
  flashcards.ts        Flashcard type and spaced-repetition scheduling
  notebooks.ts         Notebook list storage, per-notebook keys and migration of older data
  storage.ts           localStorage helpers and usePersistentState
  sources.ts           Accepted upload types and client-side file checks
  theme.ts, utils.ts   Light/dark theme; plural and relative-time helpers
  index.css            Design tokens and component styles (Tailwind utilities override them)
```

## Checks

```bash
npm run build         # type-check and production build
npm run lint          # oxlint
npm run format:check  # Prettier (npm run format to fix)
```
