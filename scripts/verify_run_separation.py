import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
data = json.loads((root / "evidence/run_separation.json").read_text())
runs = data["qwen35_runs"]
assert {run["steps"] for run in runs} == {60, 87, 178}
assert len({run["id"] for run in runs}) == 3
assert next(run for run in runs if run["steps"] == 60)["score_source"] is True
print("Qwen35 run separation: PASS")

