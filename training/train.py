from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, with_overrides
from .data import build_splits
from .rewards import (
    correct_reasoning_reward,
    exact_answer_reward,
    format_reward,
    official_solution_alignment_reward,
)
from .runtime import runtime_info, sha256_file, write_json


def _dtype(name: str):
    import torch

    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def run(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.utils import create_model_from_path

    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires a CUDA GPU")

    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    lora_config = config["lora"]
    model_name = model_config["name_or_path"]
    compute_dtype = _dtype(model_config.get("compute_dtype", "bfloat16"))
    if compute_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU/runtime does not support bfloat16; select float16")

    set_seed(int(data_config.get("seed", 42)))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=model_config.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def token_length(messages: list[dict[str, str]]) -> int:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return len(input_ids)

    train_rows, eval_rows, data_stats = build_splits(data_config, token_length=token_length)
    train_dataset = Dataset.from_list(train_rows)

    quantization_config = None
    if model_config["quantization"] == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_adapter_path = model_config.get("init_adapter_path")
    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_config.get("num_train_epochs", 1.0)),
        max_steps=int(train_config.get("max_steps", -1)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
        steps_per_generation=int(train_config.get("steps_per_generation", 1)),
        learning_rate=float(train_config.get("learning_rate", 1e-5)),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.0)),
        optim=train_config.get("optim", "paged_adamw_8bit"),
        max_grad_norm=float(train_config.get("max_grad_norm", 1.0)),
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        tf32=bool(train_config.get("tf32", False)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(train_config.get("logging_steps", 1)),
        logging_first_step=True,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=int(train_config.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(train_config.get("dataloader_pin_memory", False)),
        ddp_find_unused_parameters=bool(train_config.get("ddp_find_unused_parameters", False)),
        seed=int(data_config.get("seed", 42)),
        data_seed=int(data_config.get("seed", 42)),
        num_generations=int(train_config.get("num_generations", 2)),
        max_completion_length=int(train_config.get("max_completion_tokens", 64)),
        temperature=float(train_config.get("temperature", 0.9)),
        top_p=float(train_config.get("top_p", 0.95)),
        beta=float(train_config.get("beta", 0.0)),
        reward_weights=list(train_config.get("reward_weights", [1.0, 0.2, 0.4])),
        scale_rewards=train_config.get("scale_rewards", "none"),
        loss_type=train_config.get("loss_type", "dr_grpo"),
        mask_truncated_completions=bool(train_config.get("mask_truncated_completions", True)),
        generation_kwargs=dict(train_config.get("generation_kwargs") or {}),
        log_completions=bool(train_config.get("log_completions", True)),
        use_vllm=bool(train_config.get("use_vllm", False)),
        vllm_mode=train_config.get("vllm_mode", "colocate"),
        vllm_server_base_url=train_config.get("vllm_server_base_url"),
        vllm_server_timeout=float(train_config.get("vllm_server_timeout", 600.0)),
        vllm_gpu_memory_utilization=float(train_config.get("vllm_gpu_memory_utilization", 0.3)),
        vllm_importance_sampling_correction=bool(
            train_config.get("vllm_importance_sampling_correction", True)
        ),
        vllm_importance_sampling_mode=train_config.get(
            "vllm_importance_sampling_mode", "sequence_mask"
        ),
        model_init_kwargs=(
            None
            if initial_adapter_path
            else {
                "dtype": compute_dtype,
                "attn_implementation": model_config.get("attn_implementation", "sdpa"),
                "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
            }
        ),
    )
    peft_config = LoraConfig(
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("alpha", 16)),
        lora_dropout=float(lora_config.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_config["target_modules"]),
    )

    trainer_model: Any = model_name
    trainer_peft_config: LoraConfig | None = peft_config
    trainer_quantization_config = quantization_config
    if initial_adapter_path:
        base_model = create_model_from_path(
            model_name,
            dtype=compute_dtype,
            quantization_config=quantization_config,
            device_map=None,
            attn_implementation=model_config.get("attn_implementation", "sdpa"),
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
        trainer_model = PeftModel.from_pretrained(
            base_model,
            initial_adapter_path,
            is_trainable=True,
            autocast_adapter_dtype=False,
        )
        trainer_peft_config = None
        trainer_quantization_config = None

    torch.cuda.reset_peak_memory_stats()
    reward_funcs = [exact_answer_reward, format_reward, correct_reasoning_reward]
    if train_config.get("use_official_solution_reward", False):
        reward_funcs.append(official_solution_alignment_reward)
    trainer = GRPOTrainer(
        model=trainer_model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        quantization_config=trainer_quantization_config,
        peft_config=trainer_peft_config,
    )
    trainer.model.config.use_cache = False
    trainer.model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=train_config.get("resume_from_checkpoint"))
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_adapter)

    signal_rows = [row for row in trainer.state.log_history if "reward" in row]
    signal = {
        "logged_steps": len(signal_rows),
        "nonzero_reward_std_steps": sum(float(row.get("reward_std") or 0.0) > 0 for row in signal_rows),
        "nonzero_gradient_steps": sum(float(row.get("grad_norm") or 0.0) > 0 for row in signal_rows),
    }
    has_learning_signal = signal["nonzero_reward_std_steps"] > 0 and signal["nonzero_gradient_steps"] > 0
    manifest = {
        "status": "completed" if has_learning_signal else "completed_without_learning_signal",
        "model": model_name,
        "initial_adapter": (
            {
                "path": str(initial_adapter_path),
                "sha256": sha256_file(Path(initial_adapter_path) / "adapter_model.safetensors"),
            }
            if initial_adapter_path
            else None
        ),
        "quantization": model_config["quantization"],
        "epochs_requested": float(train_config.get("num_train_epochs", 1.0)),
        "train_metrics": result.metrics,
        "training_signal": signal,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "data": data_stats,
        "eval_records_reserved": len(eval_rows),
        "runtime": runtime_info(),
        "corpus_files": [
            {"path": source["path"], "sha256": sha256_file(source["path"])}
            for source in data_config["sources"]
            if source["kind"] == "jsonl"
        ],
        "curriculum_context_file": (
            {
                "path": data_config["curriculum_context_path"],
                "sha256": sha256_file(data_config["curriculum_context_path"]),
            }
            if data_config.get("curriculum_context_path")
            else None
        ),
        "config": config,
        "final_adapter": str(final_adapter),
    }
    if trainer.is_world_process_zero():
        write_json(output_dir / "run_manifest.json", manifest)
    trainer.accelerator.wait_for_everyone()
    if not has_learning_signal:
        raise RuntimeError(
            "the epoch completed but every GRPO group had zero reward variance or zero gradient; "
            f"see {output_dir / 'run_manifest.json'}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Suneung math model with QLoRA and GRPO.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    args = parser.parse_args()
    config = with_overrides(
        load_config(args.config),
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    manifest = run(config)
    print(json.dumps({"status": manifest["status"], "final_adapter": manifest["final_adapter"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
