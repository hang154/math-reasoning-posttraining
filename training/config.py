from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("model", "data", "training", "lora"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration section: {section}")

    model = config["model"]
    if not model.get("name_or_path"):
        raise ValueError("model.name_or_path is required")
    if model.get("quantization") not in {"none", "4bit"}:
        raise ValueError("model.quantization must be 'none' or '4bit'")

    sources = config["data"].get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("data.sources must contain at least one source")
    for source in sources:
        if source.get("kind") == "jsonl" and not source.get("path"):
            raise ValueError("a jsonl source requires path")
        if source.get("kind") == "generator" and not source.get("hook"):
            raise ValueError("a generator source requires hook")
        if source.get("kind") not in {"jsonl", "generator"}:
            raise ValueError(f"unsupported data source kind: {source.get('kind')!r}")

    stage = str(config["training"].get("stage", "grpo"))
    if stage not in {"grpo", "sft"}:
        raise ValueError("training.stage must be 'grpo' or 'sft'")
    if stage == "grpo":
        generations = int(config["training"].get("num_generations", 0))
        if generations < 2:
            raise ValueError("training.num_generations must be at least 2 for GRPO")


def with_overrides(
    config: dict[str, Any],
    *,
    output_dir: str | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    if output_dir is not None:
        updated["training"]["output_dir"] = output_dir
    if max_train_samples is not None:
        updated["data"]["max_train_samples"] = max_train_samples
    if max_eval_samples is not None:
        updated["data"]["max_eval_samples"] = max_eval_samples
    validate_config(updated)
    return updated
