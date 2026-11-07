# Running the live demo

The demo connects the frontend's **Practice MCQ** mode to the real ClearMed models. The
**Flashcards** mode is a local revision aid built from the practice questions; it is not part
of the Part II system and makes no model calls.

## What runs

| Step | Model |
|---|---|
| Answer the MCQ (A-D) and explain it | Part I FLAN-T5-base + LoRA (`checkpoints/flan_t5_lora/flan-base-full-seed42/best`) |
| Find evidence | The embedding model picked in the frontend's search settings (MiniLM, E5-base-v2 or BioLORD-2023) |
| Decide whether to answer | Frozen top-1 score gate (`configs/retrieval/gate_thresholds.json`), unless `--gate-off` |
| Write the follow-up reply | Plain FLAN-T5-base (adapter off), up to 3 passages, the frozen RQ4 settings |

The question bank is the 30 frozen demo questions in `configs/demo/demo_questions.json`
(validation questions that are not in the RQ3/RQ4 evaluation sets). The retrieval query uses the
model's **predicted** answer, never the correct one. The correct answer is sent to the browser only
so the card can reveal it after the student checks their answer.

## Start it

Install the locked dependencies first: `uv sync --frozen` from the repository root
and `npm ci` from `frontend/`. Python 3.11 is the documented training/runtime setup.

Before loading models, check the local files from the repository root:

```bash
uv run python scripts/serve_demo.py --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best --check-artifacts
```

This reports missing or empty checkpoint, tokenizer, corpus, index and configuration
files together, and checks configuration JSON syntax. It reads the passage filename
from the corpus configuration. It does not download, regenerate or modify artifacts.
Use `--corpus-run` and `--index-run` if your existing artifacts use different run paths.
Normal startup runs the same preflight before loading models.

Passing this check does **not** verify model weights, corpus checksums, embedding
compatibility or Hugging Face model/dataset caches. Those are checked or loaded by
the existing loaders during normal startup. Restore the completed experiment artifacts
when files are missing; do not retrain just to start the demo.

Terminal 1, from the repository root (needs the checkpoint, the retrieval indexes and the cached
MedMCQA dataset):

```bash
uv run python scripts/serve_demo.py --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best
```

Loading takes about a minute. Wait for `Uvicorn running on http://127.0.0.1:8000`.

After startup, `GET /api/health` returns `status: "ready"`, the question count,
available frontend embedding-model identifiers and whether the gate is enabled.
This endpoint does not run inference or return local paths. The server only starts
listening after its models, indexes and question bank load successfully; a connection
failure while loading is not a ready response. Readiness does not assess reply quality.

Terminal 2:

```bash
cd frontend
npm run dev     # PowerShell: npm.cmd run dev
```

Open the address Vite prints and switch to **Practice MCQ**.

The frontend always calls the backend through the Vite `/api` proxy. Set `CLEARMED_BACKEND_URL`
in `frontend/.env.local` (copy `frontend/.env.example`) if the backend uses another host or
port, such as `http://127.0.0.1:8001`; do not append `/api`. Restart Vite after changing it.

## Check the complete connection

1. Open `http://127.0.0.1:8000/api/health` after backend startup and confirm `status`
   is `ready`. Then open `/api/health` on the Vite origin to check the proxy too.
2. In **Practice MCQ**, the question list loads from the backend. If it shows
   "Cannot reach the tutor backend", the proxy or backend is not reachable.
3. Select a question and check an answer. The answer key is shown immediately;
   the model prediction and explanation load separately. A wrong model prediction
   is labelled as differing from the answer key. Refreshing preserves the student's
   answer and reloads the model explanation.
4. Send a follow-up, then expand its retrieved passages. Verify passage identifiers,
   text and similarity scores. Change the embedding model or top-k in search settings
   and send another turn. Only completed conversation turns enter later history.
5. A low-confidence result shows **Insufficient evidence**, the fixed fallback, and
   any returned passages. A network/model error instead offers **Retry follow-up**,
   which resends the original message, history and search settings without adding
   an error message to tutor history. Switching questions resets that conversation.

An empty question bank has an explicit empty state and retry. Invalid response data
is rejected rather than replaced with placeholder answers. Missing explanations are stated
explicitly, without inventing text.

## Troubleshooting

| Symptom | Check |
|---|---|
| Artifact preflight fails | Restore the existing checkpoint, corpus and three index runs, or select their existing paths. See the reported filenames. |
| Preflight passes but startup fails | Read the loader error: files may be incompatible/corrupt or the pretrained model/dataset caches may be unavailable. File presence alone does not prove readiness. |
| Backend health works but Vite `/api/health` fails | Check `CLEARMED_BACKEND_URL` and restart Vite. |
| Cannot reach backend / proxy error | Wait for startup to finish, verify the backend port and inspect both terminal logs. Nothing is substituted. |
| Prediction returns MALFORMED | The model failed to produce A–D; retry is available, and the answer key is never substituted as a prediction. |

Production hosting must route same-origin `/api` requests to FastAPI; the Vite
development proxy is not included in static build output.

## Regression checks

From the repository root:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run pytest -v
uv run ruff check .
git diff --check
```

From `frontend/`:

```bash
npm run test
npm run lint
npm run build
```

The API and UI suites share `tests/fixtures/demo_api.json`, a **synthetic** contract
fixture. FastAPI verifies its responses against this fixture; the UI tests consume
it through the HTTP adapter for the complete question → prediction → follow-up →
evidence flow. Other tests cover retries, malformed data, saved answers, fallbacks,
history and late responses. These tests do not load trained models or prove model
quality, network deployment or browser rendering. Complete the manual connection
checks above once the trained artifacts and caches are available.

## Gate on or off

With the frozen gate (the default, as in the proposal), the tutor often replies with the fixed
fallback message: on the 30 demo questions it answered 11 (MiniLM), 6 (E5) and 8 (BioLORD) first
follow-ups. To show a reply for every follow-up, start the server with `--gate-off`; those replies
end with a note saying the confidence check was off.

## Notes

- Everything runs on CPU. An MCQ prediction takes about 1-2 s; a follow-up reply a few seconds.
- The top-k setting controls how many retrieved passages are shown. The reply model always reads
  the top 3, as in RQ4.
- If the Hugging Face cache is offline, prefix the command with `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`.
