# Running the live demo

The demo connects the frontend's **Practice MCQ** mode to the real ClearMed models. Notebook and
Flashcards modes stay as labelled previews; they are not part of the Part II system.

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
# PowerShell:  $env:VITE_USE_MOCK="false"; npm.cmd run dev
VITE_USE_MOCK=false npm run dev
```

Open the address Vite prints and switch to **Practice MCQ**.

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
