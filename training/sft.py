from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, with_overrides
from .data import build_splits
from .runtime import runtime_info, sha256_file, write_json
from .train import _dtype


def build_supervised_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    reference = str(row.get("reference_solution") or "").strip()
    if not reference:
        raise ValueError(f"missing reference solution for {row.get('item_id')}")
    # Qwen's thinking template otherwise inserts an empty <think> block and
    # places the official solution in ordinary assistant content. Put the EBS
    # trace inside the reasoning block explicitly so assistant-only SFT trains
    # the same channel used by thinking-enabled inference and GRPO rollouts.
    target = f"<think>\n{reference}\n</think>\n<answer>{row['answer']}</answer>"
    return [*row["prompt"], {"role": "assistant", "content": target}]


def run(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("SFT training requires a CUDA GPU")
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    lora_config = config["lora"]
    model_name = model_config["name_or_path"]
    compute_dtype = _dtype(model_config.get("compute_dtype", "bfloat16"))
    set_seed(int(data_config.get("seed", 42)))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=model_config.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def token_length(messages: list[dict[str, str]]) -> int:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True)
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return len(input_ids)

    train_rows, eval_rows, data_stats = build_splits(data_config, token_length=token_length)
    train_dataset = Dataset.from_list(
        [{"messages": build_supervised_messages(row), "item_id": row["item_id"]} for row in train_rows]
    )
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_config.get("num_train_epochs", 1.0)),
        max_steps=int(train_config.get("max_steps", -1)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_config.get("learning_rate", 1e-5)),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.0)),
        optim=train_config.get("optim", "adamw_torch_fused"),
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
        max_length=int(train_config.get("max_length", 4096)),
        packing=False,
        assistant_only_loss=True,
        loss_type=train_config.get("loss_type", "nll"),
        model_init_kwargs={
            "dtype": compute_dtype,
            "attn_implementation": model_config.get("attn_implementation", "sdpa"),
            "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        },
    )
    peft_config = LoraConfig(
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("alpha", 16)),
        lora_dropout=float(lora_config.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_config["target_modules"]),
    )

    torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(
        model=model_name,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.config.use_cache = False
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_adapter)

    manifest = {
        "status": "completed",
        "stage": "official_solution_sft",
        "model": model_name,
        "train_metrics": result.metrics,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "data": data_stats,
        "eval_records_reserved": len(eval_rows),
        "runtime": runtime_info(),
        "corpus_files": [
            {"path": source["path"], "sha256": sha256_file(source["path"])}
            for source in data_config["sources"]
            if source["kind"] == "jsonl"
        ],
        "config": config,
        "final_adapter": str(final_adapter),
    }
    if trainer.is_world_process_zero():
        write_json(output_dir / "run_manifest.json", manifest)
    trainer.accelerator.wait_for_everyone()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT on official EBS solution traces.")
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
