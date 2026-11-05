"""Serve the live ClearMed demo for the frontend's Practice MCQ mode (Plan 8).

Loads Regan's frozen FLAN-T5 + LoRA checkpoint (MCQ answer), the three
retrieval indexes and frozen gate thresholds, and the RQ4 reply settings
(plain FLAN-T5 with the adapter off, cleaned evidence, no-repeat-3), then
serves http://localhost:8000. Start the frontend with ``VITE_USE_MOCK=false``;
Vite proxies ``/api`` here.

    uv run python scripts/serve_demo.py --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best

``--gate-off`` answers every follow-up from its retrieved evidence and labels
those replies; the default keeps the frozen gate, as in the proposal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clearmed.conversation.pipeline import GATE_OFF_SUFFIX, TutorPipeline
from clearmed.retrieval.embedders import METHOD_IDS

DEMO_PATH = Path("configs/demo/demo_questions.json")
FROZEN_PATH = Path("configs/conversation/rq4_frozen.json")


def load_demo_questions(validation: Any, demo_path: Path = DEMO_PATH) -> list[dict[str, Any]]:
    """The frozen demo rows, in the order the config lists them."""
    from clearmed.app.demo import question_from_row

    ids = [str(i) for i in json.loads(demo_path.read_text("utf-8"))["question_ids"]]
    rows = {str(row["id"]): row for row in validation if str(row["id"]) in set(ids)}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise ValueError(f"{len(missing)} demo questions are not in the validation split, e.g. {missing[:3]}.")
    return [question_from_row(rows[i]) for i in ids]


def build_service(args: argparse.Namespace) -> Any:
    import torch

    from clearmed.app.demo import TutorService
    from clearmed.conversation.pipeline import DenseRetriever, FlanT5Generator
    from clearmed.data.loader import load_medmcqa_split
    from clearmed.retrieval.corpus import load_corpus
    from clearmed.retrieval.embedders import (
        Embedder,
        SentenceTransformerEncoder,
        load_embedding_config,
    )
    from clearmed.retrieval.gate import Gate
    from clearmed.retrieval.index import DenseIndex
    from clearmed.training.flan_t5_lora import FlanT5Predictor, load_adapter_checkpoint

    frozen = json.loads(FROZEN_PATH.read_text("utf-8"))
    print(f"Reply model (frozen {frozen['frozen_on']}): {frozen['reply_model']['decoding']}, adapter off.")
    device = torch.device(args.device)
    model, tokenizer, data_config, _ = load_adapter_checkpoint(args.adapter_dir, device)
    predictor = FlanT5Predictor(model, tokenizer, data_config, device)
    generator = FlanT5Generator(model, tokenizer, max_input_tokens=data_config.max_source_length, device=device,
                                no_repeat_ngram_size=3, use_adapter=False)
    passages, _ = load_corpus(args.corpus_run)
    tutors: dict[str, TutorPipeline] = {}
    for method_id in METHOD_IDS:
        config = load_embedding_config(method_id)
        retriever = DenseRetriever(
            Embedder(config, SentenceTransformerEncoder(config)),
            DenseIndex.load(Path("outputs/retrieval") / method_id / args.index_run),
            passages,
        )
        if args.gate_off:
            tutors[method_id] = TutorPipeline(generator, stage="demo", retriever=retriever, gate_off=True,
                                              method_id=f"{method_id}{GATE_OFF_SUFFIX}", clean_evidence=True)
        else:
            tutors[method_id] = TutorPipeline(generator, stage="demo", retriever=retriever, gate=Gate.load(method_id),
                                              method_id=method_id, clean_evidence=True)
        print(f"Loaded {method_id}.")
    questions = load_demo_questions(load_medmcqa_split("validation"))
    return TutorService(questions, predict=lambda row: predictor.generate_batch([row])[0], tutors=tutors,
                        passages={str(p["doc_id"]): p for p in passages}, gate_off=args.gate_off)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", type=Path, required=True, help="Frozen FLAN-T5 + LoRA adapter directory.")
    parser.add_argument("--index-run", default="full-20261016")
    parser.add_argument("--corpus-run", type=Path, default=Path("outputs/retrieval/corpus/20261016-142815"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate-off", action="store_true", help="Answer every follow-up; replies are labelled.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    from clearmed.app.demo import create_app

    app = create_app(build_service(args))
    print(f"ClearMed demo on http://{args.host}:{args.port} (gate {'off' if args.gate_off else 'on'}).")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
