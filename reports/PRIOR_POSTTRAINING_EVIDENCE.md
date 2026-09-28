# Prior post-training evidence audit

Generated: 2026-09-28

This audit separates results that came from different runs. In particular, the
87-step GRPO run is **not** the run that produced the later 99 → 103 → 106
evaluation table.

## Qwen3.6-35B-A3B: portfolio-ready evidence

### High-difficulty Korean mathematics SFT → GRPO

Status: **VERIFIED**

- Model path recorded by the run manifests: `Qwen3.6-35B-A3B`.
- Controlled high-v2 corpus: 535 eligible unique Korean high-school four-point
  mathematics items; the SFT and GRPO manifests record 534 train and one eval
  item.
- SFT manifest: completed; 326.5622 seconds, loss 0.741616, peak allocation
  85,892,577,280 bytes. The output path and manifest identify the 4-GPU SFT
  variant.
- Matched high-v2 GRPO manifest: completed; 60 logged steps, all 60 with
  non-zero reward standard deviation and gradients; 3,629.4258 seconds, peak
  allocation 140,935,469,568 bytes.
- Evaluation on the fixed 128-item hard set:

| Stage | Aggregate | MMLU-Pro subset | MuSR subset |
|---|---:|---:|---:|
| Base | 99/128 | 60/64 | 39/64 |
| high-v2 SFT | 103/128 | 59/64 | 44/64 |
| high-v2 GRPO | 106/128 | 61/64 | 45/64 |

- On a separate 61-item unseen BBEH set under the documented Qwen-recommended
  64k-thinking sampling policy: base 33/61, high-v2 SFT 36/61, high-v2 GRPO
  41/61. The report explicitly limits this to a single seed and reports
  base→GRPO McNemar `p=0.3018`; it must not be presented as statistically
  conclusive general reasoning improvement.
- A separate 32k training line is also verified: 535 eligible items, 534 train
  items, 178 logged GRPO steps, 23,011.0146 seconds, and 5,745,115 cumulative
  tokens in the final trainer state. This is operational evidence, not the
  source of the 60-step high-v2 score table above.

Primary evidence:

- `sources/qwen35/high_v2_sft_4gpu_run_manifest.json`
- `sources/qwen35/high_v2_grpo60_run_manifest.json`
- `sources/qwen35/high_v2_comparison.json`
- `sources/qwen35/BBEH_UNSEEN61_THINKING64K_QWEN_RECOMMENDED_20260809.md`
- `sources/qwen35/high_v2_32k_grpo_run_manifest.json`
- `sources/qwen35/high_v2_32k_checkpoint178_trainer_state.json`

### Earlier 87-step Qwen GRPO run

Status: **VERIFIED AS A SEPARATE RUN**

- Completed 87 optimizer steps over 261 train samples drawn from 326 eligible
  Math I/II 3- and 4-point items.
- Runtime: 4,310.6635 seconds; peak allocation: 123,947,326,464 bytes.
- All 87 logged steps had non-zero reward standard deviation and non-zero
  gradients.
- LoRA configuration: rank 16, alpha 32.
- The often-quoted 783 sampled completions is derivable as 261 samples × three
  generations, but the preserved manifest does not itself record a completion
  count. Keep it labelled as derived unless the original generation ledger is
  recovered.
- The often-quoted ~2.56M completion tokens was not found in the preserved
  manifest or compact logs during this audit. Do not use it as verified.
- “4.79 H200 GPU-hours” equals runtime × four GPUs, but the preserved manifest
  does not record world size. Do not use it as verified until the launcher or
  telemetry record is recovered.

Primary evidence:

- `sources/qwen35/earlier_87step_grpo_run_manifest.json`
- `sources/qwen35/earlier_87step_adapter_config.json`

## Qwen3-0.6B controlled SFT → GRPO

Status: **VERIFIED**

The same fixed GSM8K-100 evaluation file records:

| Stage | Exact numeric accuracy | Required-format rate |
|---|---:|---:|
| Base | 22% | 0% |
| SFT | 36% | 45% |
| GRPO | 47% | 75% |

Primary evidence: `sources/qwen06/math_eval_100.json`.

## Nemotron Nano 12B v2

Status: **TRAINING VERIFIED; OUTCOME CLAIMS NOT YET VERIFIED**

- A completed SFT manifest exists for `NVIDIA-Nemotron-Nano-12B-v2-Base` on
  the same 535-item high-v2 corpus: 534 train items, one eval item, 803.0901
  seconds, loss 1.058024, peak allocation 97,289,364,480 bytes.
- Durable GRPO checkpoint-status files exist at step 60 for high-v2 and for a
  later 32k-matched variant.
- The preserved tree contains multiple completed checkpoint markers, but this
  audit did not find a compact, attributable base/SFT/GRPO benchmark comparison.
  Therefore Nemotron can currently support the claim “ran SFT and GRPO jobs and
  retained checkpoints,” but not a performance-improvement claim.

Primary evidence:

- `sources/nemotron12b/sft_run_manifest.json`
- `sources/nemotron12b/high_v2_grpo_checkpoint_status.json`
- `sources/nemotron12b/high_v2_qwen32k_grpo_checkpoint_status.json`

## Qwen3.8-27B inference and data-system operations

Status: **VERIFIED AS SYSTEMS EVIDENCE, NOT TRAINING**

- A two-backend router was exercised with 12 concurrent requests split 6/6.
- A later GPU3-primary policy passed 8/8 direct concurrent requests and 24/24
  requests through the stable agent endpoint, with GPU2 spillover after eight
  active short requests and GPU2-only routing above the GPU3 context limit.
- This supports model-serving, routing, concurrency, failure fallback, and
  evaluation/data-pipeline experience. It must not be described as Qwen3.8
  training or fine-tuning.

Primary evidence:

- `sources/qwen38_ops/GPU23_BALANCED_ROUTER_20260928.md`
- `sources/qwen38_ops/GPU3_PRIMARY_AND_PEAK_SPILLOVER_20260928.md`

## Safe résumé ordering

1. Qwen3.6-35B-A3B controlled high-v2 SFT/GRPO and fixed evaluations.
2. 1.012B scratch pretraining and optimizer experiments from the current H200
   portfolio.
3. Qwen3-0.6B fixed-holdout SFT/GRPO reproduction.
4. Video post-training with explicit regression analysis.
5. Nemotron 12B execution evidence, without an improvement number until its
   benchmark outputs are recovered.
6. Qwen3.8-27B serving/data operations as supporting systems experience.

## Claim boundary

The records establish direct execution, checkpoint production, evaluation, and
systems operation. They do not establish years of senior production ownership.
Do not merge statistics from the 87-step run, 60-step high-v2 run, and 178-step
32k run into one experiment.
