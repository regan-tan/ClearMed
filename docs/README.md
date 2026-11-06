# Documentation

The final report has the method and results. These pages hold the per-model and per-step detail.
Numbers in the Part I model pages come from development and smoke runs on `internal_dev`. The
final held-out results are in the report and the [root README](../README.md).

## Part I: MCQ answering and explanation (RQ1, RQ2)

| Page | What it covers |
|---|---|
| [tfidf_baseline.md](tfidf_baseline.md) | TF-IDF + logistic regression baseline |
| [seq2seq.md](seq2seq.md) | Scratch GRU Seq2Seq: preprocessing, training, smoke results |
| [flan_t5_lora.md](flan_t5_lora.md) | FLAN-T5 + LoRA: data contract, training and evaluation |
| [flan_t5_gpu_training.md](flan_t5_gpu_training.md) | Running FLAN-T5 + LoRA training on the SMU Origami GPU cluster |

## Part II: retrieval and grounded follow-ups (RQ3, RQ4)

| Page | What it covers |
|---|---|
| [retrieval_labelling_guideline.md](retrieval_labelling_guideline.md) | How retrieved passages were labelled relevant or not (RQ3, thresholds) |
| [rq4_grading_guide.md](rq4_grading_guide.md) | Rubric and worked examples used to score RQ4 replies |
| [demo.md](demo.md) | Running the live demo (backend + frontend Practice MCQ mode) |

## Both parts

| Page | What it covers |
|---|---|
| [architecture/README.md](architecture/README.md) | System overview and pipeline diagrams |
| [project-proposal.md](project-proposal.md) | The original project proposal (historical; the report has the final design) |
