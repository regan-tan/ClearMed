import json

import pytest

from clearmed.app.startup import check_demo_artifacts


@pytest.fixture
def artifacts(tmp_path):
    adapter = tmp_path / "adapter"
    corpus = tmp_path / "corpus"
    index = tmp_path / "index"
    for directory in (adapter, corpus, index):
        directory.mkdir()
    for name in ("adapter_metadata.json", "adapter_config.json", "tokenizer_config.json", "tokenizer.json"):
        (adapter / name).write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic weights")
    (corpus / "config.json").write_text(json.dumps({"passages_file": "custom.jsonl"}))
    (corpus / "custom.jsonl").write_text('{}\n')
    (index / "index.json").write_text("{}")
    (index / "doc_ids.json").write_text('["d1"]')
    (index / "embeddings.npy").write_bytes(b"synthetic embeddings")
    config = tmp_path / "demo.json"
    config.write_text("{}")
    return {"adapter_dir": adapter, "corpus_run": corpus, "index_dirs": {"minilm": index}, "config_paths": [config]}


def test_preflight_accepts_present_files_without_loading_or_modifying_them(artifacts):
    root = artifacts["adapter_dir"].parent
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    check_demo_artifacts(**artifacts)
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_preflight_reports_all_missing_artifacts_together(artifacts):
    missing = [artifacts["adapter_dir"] / "adapter_metadata.json",
               artifacts["corpus_run"] / "custom.jsonl",
               artifacts["index_dirs"]["minilm"] / "embeddings.npy"]
    for path in missing:
        path.unlink()
    with pytest.raises(ValueError) as error:
        check_demo_artifacts(**artifacts)
    assert all(str(path) in str(error.value) for path in missing)
    assert "--adapter-dir" in str(error.value)


@pytest.mark.parametrize("content", ["not json", "[]", '{"passages_file": 42}', '{}'])
def test_preflight_rejects_invalid_corpus_configuration(artifacts, content):
    (artifacts["corpus_run"] / "config.json").write_text(content)
    with pytest.raises(ValueError, match="config.json"):
        check_demo_artifacts(**artifacts)


def test_preflight_accepts_legacy_weights_and_sentencepiece_tokenizer(artifacts):
    adapter = artifacts["adapter_dir"]
    (adapter / "adapter_model.safetensors").rename(adapter / "adapter_model.bin")
    (adapter / "tokenizer.json").rename(adapter / "spiece.model")
    check_demo_artifacts(**artifacts)


def test_preflight_rejects_empty_weights_and_tokenizer(artifacts):
    adapter = artifacts["adapter_dir"]
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    (adapter / "tokenizer.json").write_bytes(b"")
    with pytest.raises(ValueError) as error:
        check_demo_artifacts(**artifacts)
    assert "adapter_model" in str(error.value) and "tokenizer.json" in str(error.value)
