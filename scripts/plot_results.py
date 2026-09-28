#!/usr/bin/env python3
"""Generate the README SVG directly from the audited comparison JSON."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "sources/qwen35/high_v2_comparison.json").read_text())["scores"]
keys = ["base", "high_v2_sft", "high_v2_grpo"]
labels = ["Base", "SFT", "GRPO"]
hard = [data[key]["hard"]["correct"] for key in keys]
musr = [data[key]["hard"]["by_dataset"]["musr"]["correct"] for key in keys]
colors = ["#64748b", "#2563eb", "#16a34a"]
bars = []
for i, (label, value, color) in enumerate(zip(labels, hard, colors)):
    x = 160 + i * 180
    height = value * 2.2
    y = 300 - height
    bars.append(f'<rect x="{x}" y="{y:.1f}" width="100" height="{height:.1f}" rx="8" fill="{color}"/>')
    bars.append(f'<text x="{x+50}" y="{y-10:.1f}" text-anchor="middle" font-size="20" font-weight="700">{value}/128</text>')
    bars.append(f'<text x="{x+50}" y="330" text-anchor="middle" font-size="18">{label}</text>')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="400" viewBox="0 0 760 400">
<rect width="760" height="400" fill="#f8fafc"/><text x="40" y="45" font-size="26" font-weight="700">Qwen3.6-35B-A3B — fixed Hard128</text>
<text x="40" y="75" font-size="15" fill="#475569">MuSR subset: 39 → 44 → 45 · BBEH61: 33 → 36 → 41 (single seed, p=0.3018)</text>
<line x1="100" y1="300" x2="700" y2="300" stroke="#94a3b8"/>{''.join(bars)}</svg>'''
(ROOT / "plots").mkdir(exist_ok=True)
(ROOT / "plots/qwen35_hard128.svg").write_text(svg)
print(ROOT / "plots/qwen35_hard128.svg")
