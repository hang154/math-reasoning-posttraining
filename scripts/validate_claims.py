from pathlib import Path


root = Path(__file__).resolve().parents[1]
text = "\n".join(p.read_text(errors="ignore") for p in root.rglob("*.md"))
assert "p=0.3018" in text
assert "not presented as statistically conclusive" in text
assert "2.56M completion" in text and "4.79 GPU-hours were not verified" in text
print("math post-training public-claim validation: PASS")
