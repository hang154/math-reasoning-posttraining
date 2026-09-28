from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


STRICT_REASONING_RE = re.compile(
    r"^\s*<answer>\s*(?P<answer>.+?)\s*</answer>\s*"
    r"<reasoning>\s*(?P<reasoning>.+?)\s*</reasoning>\s*$",
    re.DOTALL,
)
STRICT_ANSWER_ONLY_RE = re.compile(r"^\s*<answer>\s*(?P<answer>.+?)\s*</answer>\s*$", re.DOTALL)
STRICT_QWEN_THINKING_RE = re.compile(
    r"^\s*(?:<think>\s*)?(?P<reasoning>.+?)(?:\s*</think>)?\s*"
    r"<answer>\s*(?P<answer>.+?)\s*</answer>\s*$",
    re.DOTALL,
)
ANSWER_RE = re.compile(r"<answer>\s*(?P<answer>.+?)\s*</answer>", re.DOTALL)
CIRCLED = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}
SOLUTION_TOKEN_RE = re.compile(
    r"\\[A-Za-z]+|[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?|"
    r"-?\d+(?:\.\d+)?|[가-힣]{2,}|[=<>+\-×÷^]"
)
SOLUTION_STOPWORDS = {
    "그리고",
    "그러므로",
    "따라서",
    "이므로",
    "이면",
    "에서",
    "정답",
    "풀이",
    "출제의도",
    "한다",
    "하면",
}


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, list):
        return "".join(completion_text(part) for part in completion)
    return str(completion)


def extract_tagged_answer(completion: Any) -> str | None:
    matches = list(ANSWER_RE.finditer(completion_text(completion)))
    return matches[-1].group("answer").strip() if matches else None


def extract_reasoning(completion: Any) -> str | None:
    text = completion_text(completion)
    match = STRICT_REASONING_RE.fullmatch(text) or STRICT_QWEN_THINKING_RE.fullmatch(text)
    return match.group("reasoning").strip() if match else None


def extract_answer_candidate(completion: Any, answer_type: str) -> str | None:
    tagged = extract_tagged_answer(completion)
    if tagged is not None:
        return tagged
    text = completion_text(completion).strip().replace("$", "")
    if answer_type in {"choice", "multiple_choice"}:
        choice_or_value = r"(?:[①②③④⑤]|[1-5](?:번)?|-?\d+(?:\.\d+)?(?:/\d+)?|\\frac\{-?\d+\}\{\d+\})"
        return text if re.fullmatch(choice_or_value, text) else None
    bare_number = r"-?\d+(?:\.\d+)?(?:/\d+)?"
    latex_fraction = r"\\frac\{-?\d+\}\{\d+\}"
    return text if re.fullmatch(f"(?:{bare_number}|{latex_fraction})", text) else None


def normalize_answer(value: str | None, answer_type: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().replace("$", "").replace(" ", "")
    normalized = re.sub(r"^\\boxed\{(.+)\}$", r"\1", normalized)
    if answer_type in {"choice", "multiple_choice"}:
        normalized = CIRCLED.get(normalized, normalized)
        match = re.fullmatch(r"(?:선택지|정답)?([1-5])(?:번)?", normalized)
        return match.group(1) if match else normalized

    normalized = normalized.replace(",", "")
    try:
        return str(Decimal(normalized).normalize())
    except InvalidOperation:
        return normalized


def exact_answer_reward(
    completions: list[Any], answer_candidates: list[list[str]], answer_type: list[str], **_: Any
) -> list[float]:
    rewards = []
    for completion, expected_values, kind in zip(completions, answer_candidates, answer_type, strict=True):
        predicted = normalize_answer(extract_answer_candidate(completion, kind), kind)
        gold_values = {normalize_answer(expected, kind) for expected in expected_values}
        rewards.append(1.0 if predicted is not None and predicted in gold_values else 0.0)
    return rewards


def correct_reasoning_reward(
    completions: list[Any], answer_candidates: list[list[str]], answer_type: list[str], **_: Any
) -> list[float]:
    """Reward a non-trivial reasoning block only when its final answer is correct."""
    rewards = []
    for completion, expected_values, kind in zip(completions, answer_candidates, answer_type, strict=True):
        reasoning = extract_reasoning(completion)
        predicted = normalize_answer(extract_answer_candidate(completion, kind), kind)
        gold_values = {normalize_answer(expected, kind) for expected in expected_values}
        has_reasoning = reasoning is not None and len(reasoning) >= 12
        rewards.append(1.0 if has_reasoning and predicted in gold_values else 0.0)
    return rewards


def _solution_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in SOLUTION_TOKEN_RE.findall(text.replace(" ", ""))
        if token not in SOLUTION_STOPWORDS and not re.fullmatch(r"[가-힣]{0,1}", token)
    }


def official_solution_alignment_reward(
    completions: list[Any],
    answer_candidates: list[list[str]],
    answer_type: list[str],
    reference_solution: list[str | None],
    **_: Any,
) -> list[float]:
    """Reward correct reasoning that covers deterministic anchors from the EBS solution."""
    rewards = []
    for completion, expected_values, kind, reference in zip(
        completions, answer_candidates, answer_type, reference_solution, strict=True
    ):
        reasoning = extract_reasoning(completion)
        predicted = normalize_answer(extract_answer_candidate(completion, kind), kind)
        gold_values = {normalize_answer(expected, kind) for expected in expected_values}
        if not reference or not reasoning or predicted not in gold_values:
            rewards.append(0.0)
            continue
        reference_tokens = _solution_tokens(reference)
        reasoning_tokens = _solution_tokens(reasoning)
        if not reference_tokens or not reasoning_tokens:
            rewards.append(0.0)
            continue
        overlap = len(reference_tokens & reasoning_tokens)
        precision = overlap / len(reasoning_tokens)
        recall = overlap / len(reference_tokens)
        rewards.append(2 * precision * recall / (precision + recall) if overlap else 0.0)
    return rewards


def is_strict_response(completion: Any, response_format: str) -> bool:
    text = completion_text(completion)
    if response_format == "answer_only":
        return STRICT_ANSWER_ONLY_RE.fullmatch(text) is not None
    if response_format == "qwen_thinking":
        return STRICT_QWEN_THINKING_RE.fullmatch(text) is not None
    return STRICT_REASONING_RE.fullmatch(text) is not None


def format_reward(
    completions: list[Any], response_format: list[str], answer_type: list[str] | None = None, **_: Any
) -> list[float]:
    rewards = []
    for index, (completion, expected_format) in enumerate(zip(completions, response_format, strict=True)):
        text = completion_text(completion)
        kind = answer_type[index] if answer_type is not None else "short_answer"
        if is_strict_response(text, expected_format):
            rewards.append(1.0)
        elif ANSWER_RE.search(text) or extract_answer_candidate(text, kind) is not None:
            rewards.append(0.25)
        else:
            rewards.append(0.0)
    return rewards
