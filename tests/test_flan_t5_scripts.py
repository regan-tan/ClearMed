import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = spec_from_file_location(f"{name}_for_test", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_train_cli_supports_required_smoke_arguments():
    module = load_script("train_flan_t5_lora")
    args = module.build_parser().parse_args(["--max-train-questions", "8", "--max-dev-questions", "4", "--batch-size", "2", "--max-steps", "1", "--device", "cpu", "--run-id", "flan-smoke-v1"])
    assert args.model_name == "google/flan-t5-base"
    assert args.max_steps == 1 and args.lora_target_modules == ["q", "v"]


def test_cuda_requirement_rejects_silent_cpu_fallback(monkeypatch):
    module = load_script("train_flan_t5_lora")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    try:
        module.resolve_device("auto", require_cuda=True)
    except RuntimeError as error:
        assert "refusing" in str(error)
    else:
        raise AssertionError("GPU-required runs must not silently use CPU")


def test_train_cli_resolves_auto_precision_for_cpu_before_config_construction():
    module = load_script("train_flan_t5_lora")
    args = module.build_parser().parse_args(["--device", "cpu"])

    train_config = module._train_config_from_args(args, module.resolve_device(args.device))

    assert args.precision == "auto"
    assert train_config.precision == "fp32"


def test_train_cli_resolves_cuda_auto_and_preserves_explicit_precision(monkeypatch):
    module = load_script("train_flan_t5_lora")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "is_bf16_supported", lambda: True)
    device = module.resolve_device("cuda")

    auto_args = module.build_parser().parse_args(["--device", "cuda"])
    explicit_args = module.build_parser().parse_args(["--device", "cuda", "--precision", "fp16"])

    assert device == torch.device("cuda")
    assert module._train_config_from_args(auto_args, device).precision == "bf16"
    assert module._train_config_from_args(explicit_args, device).precision == "fp16"
