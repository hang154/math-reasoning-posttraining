#!/usr/bin/env bash
set -euo pipefail
accelerate launch --num_processes 4 -m training.grpo --config configs/qwen35_high_v2_grpo60.yaml
