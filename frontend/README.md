# ClearMed Frontend

React + TypeScript + Vite + Tailwind CSS interface for ClearMed Study.

## Run

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

By default the app uses a mock backend ([src/api/mock.ts](src/api/mock.ts)) with sample questions, so it runs without any Python services.

To use the real backend, start the FastAPI server on port 8000 and run:

```bash
VITE_USE_MOCK=false npm run dev
```

Vite proxies `/api/*` to `http://localhost:8000`.

## API contract

Types live in [src/api/types.ts](src/api/types.ts). The backend should expose:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/questions` | – | `Question[]` |
| POST | `/api/questions/{id}/predict` | – | `McqPrediction` |
| POST | `/api/follow-up` | `FollowUpRequest` | `FollowUpResponse` |

`FollowUpResponse.fallback` is `true` when retrieval found insufficient evidence; `reply` is then the fixed deterministic fallback message, not model output.

## Layout

- **Left:** question bank
- **Centre:** MCQ card (click to answer, reveals answer key and model explanation) and follow-up chat with retrieved evidence
- **Right:** embedding model and top-k selectors, upcoming study tools (flashcards, quizzes, notes)
