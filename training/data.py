from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol


SYSTEM_PROMPT = (
    "당신은 한국 고등학교 교육과정 안에서 수능 수학 문항을 푸는 조교다. "
    "계산과 논리를 점검하고 지정된 형식만 사용하라."
)
CIRCLED_CHOICE_INDEX = {"①": 0, "②": 1, "③": 2, "④": 3, "⑤": 4}


class QuestionGenerator(Protocol):
    """Contract for a future online/offline item generator adapter."""

    def __call__(self, **options: Any) -> Iterable[dict[str, Any]]: ...


@dataclass(frozen=True)
class TrainingExample:
    item_id: str
    prompt: list[dict[str, str]]
    answer: str
    answer_candidates: list[str]
    answer_type: str
    response_format: str
    source_kind: str
    curriculum_context_key: str | None
    reference_solution: str | None
    score: int | None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def load_raw_rows(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    kind = source["kind"]
    if kind == "jsonl":
        path = Path(source["path"])
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                row.setdefault("_grpo_source_kind", "corpus")
                yield row
        return

    module_name, separator, function_name = source["hook"].partition(":")
    if not separator:
        raise ValueError("generator hook must use the form 'module:function'")
    factory: QuestionGenerator = getattr(importlib.import_module(module_name), function_name)
    options = source.get("options") or {}
    for row in factory(**options):
        if not isinstance(row, dict):
            raise ValueError("generator hooks must yield dictionaries")
        row.setdefault("_grpo_source_kind", "generator")
        yield row


def normalize_row(row: dict[str, Any], data_config: dict[str, Any]) -> tuple[TrainingExample | None, str | None]:
    if "question" in row and "answer" in row:
        item_id = str(row.get("item_id") or _content_id(row))
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        answer_candidates = [str(value).strip() for value in row.get("accepted_answers", [answer])]
        answer_type = str(row.get("answer_type") or "short_answer")
        has_visual = bool(row.get("has_visual"))
        is_gold = bool(row.get("is_gold", True))
        answer_needs_review = bool(row.get("answer_needs_review", False))
        score = row.get("score")
        subject = str(row.get("subject") or "")
        content_area = str(row.get("content_area") or "")
        reference_solution = str(
            row.get("reference_solution")
            or (row.get("official_solution") or {}).get("solution_text")
            or ""
        ).strip()
    else:
        canonical = row.get("canonical") or {}
        canonical_problem = canonical.get("problem") or {}
        canonical_answer = canonical.get("answer") or {}
        deterministic = row.get("deterministic_features") or {}
        top_quality = row.get("quality") or {}
        canonical_quality = canonical.get("quality") or {}

        item_id = str(row.get("item_id") or canonical.get("problem_id") or _content_id(row))
        question = str((row.get("raw_preserved") or {}).get("ocr_markdown") or "").strip()
        answer = str(
            canonical_answer.get("answer_normalized") or canonical_answer.get("answer_raw") or ""
        ).strip()
        answer_candidates = _answer_candidates(answer, canonical_problem)
        answer_type = str(canonical_answer.get("answer_type") or canonical_problem.get("type") or "")
        has_visual = any(
            bool(deterministic.get(field)) for field in ("has_figure", "has_graph", "has_table")
        )
        is_gold = bool(
            top_quality.get("ocr_quality_gate_pass") is True
            or canonical_quality.get("ocr_gold_pass") is True
        )
        answer_needs_review = bool(canonical_answer.get("answer_needs_review", False))
        score = deterministic.get("score")
        curriculum = (row.get("manual_classification") or {}).get("curriculum") or {}
        subject = str(curriculum.get("course") or "")
        content_area = str(curriculum.get("content_area") or "")
        reference_solution = str(
            row.get("reference_solution")
            or (row.get("official_solution") or {}).get("solution_text")
            or ""
        ).strip()

    if data_config.get("require_gold", True) and not is_gold:
        return None, "not_gold"
    if answer_needs_review:
        return None, "answer_needs_review"
    if not question:
        return None, "missing_question"
    if not answer:
        return None, "missing_answer"
    if data_config.get("require_official_solution", False) and not reference_solution:
        return None, "missing_official_solution"
    if data_config.get("exclude_visual", True) and has_visual:
        return None, "visual_item"
    allowed_scores = data_config.get("allowed_scores")
    if allowed_scores is not None and (
        score is None or int(score) not in {int(value) for value in allowed_scores}
    ):
        return None, "score_not_allowed"
    if data_config.get("max_score") is not None and score is not None:
        if int(score) > int(data_config["max_score"]):
            return None, "score_above_max"

    curriculum_context, curriculum_context_key = resolve_curriculum_context(
        data_config, subject, content_area
    )
    if data_config.get("curriculum_context_path") and curriculum_context is None:
        return None, "missing_curriculum_context"

    response_format = str(data_config.get("response_format", "reasoning"))
    prompt = build_prompt(
        question,
        answer_type,
        response_format,
        reasoning_token_budget=data_config.get("reasoning_token_budget"),
        curriculum_context=curriculum_context,
    )
    return (
        TrainingExample(
            item_id=item_id,
            prompt=prompt,
            answer=answer,
            answer_candidates=answer_candidates,
            answer_type=answer_type,
            response_format=response_format,
            source_kind=str(row.get("_grpo_source_kind") or "corpus"),
            curriculum_context_key=curriculum_context_key,
            reference_solution=reference_solution or None,
            score=int(score) if score is not None else None,
        ),
        None,
    )


def build_prompt(
    question: str,
    answer_type: str,
    response_format: str = "reasoning",
    reasoning_token_budget: int | None = None,
    curriculum_context: str | None = None,
) -> list[dict[str, str]]:
    if answer_type in {"choice", "multiple_choice"}:
        final_rule = "객관식 정답은 ①~⑤ 중 하나만 답 태그 안에 쓴다."
    else:
        final_rule = "단답형 정답은 숫자 또는 기약분수만 답 태그 안에 쓴다."
    if response_format == "answer_only":
        output_rule = (
            "다른 설명 없이 정확히 <answer>정답</answer> 한 줄만 출력한다. "
            "예: <answer>③</answer>"
        )
    elif response_format == "reasoning":
        output_rule = (
            "첫 줄에 <answer>정답</answer>, 다음 줄에 "
            "<reasoning>60자 이내의 한 문장 풀이</reasoning>만 출력한다."
        )
    elif response_format == "qwen_thinking":
        budget_rule = (
            f"풀이 전체를 {int(reasoning_token_budget):,}토큰 이내로 끝낸다. "
            if reasoning_token_budget is not None
            else ""
        )
        output_rule = (
            "문제 재진술이나 메타 설명 없이 핵심 식 중심으로 간결하게 풀이하고 검산한다. "
            + budget_rule
            +
            "마지막 줄에는 "
            "정확히 <answer>정답</answer>만 출력한다. 중간 풀이에 정답 태그를 쓰지 않는다."
        )
    else:
        raise ValueError(f"unsupported data.response_format: {response_format}")
    system = SYSTEM_PROMPT
    if curriculum_context:
        system += f"\n\n[현재 문항의 교육과정 맥락]\n{curriculum_context}"
    user = f"[문항]\n{question}\n\n[출력 규칙]\n{output_rule} {final_rule}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@lru_cache(maxsize=None)
def load_curriculum_contexts(path: str) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    contexts = payload.get("contexts") or {}
    return {str(key): str(value["content"]).strip() for key, value in contexts.items()}


def resolve_curriculum_context(
    data_config: dict[str, Any], subject: str, content_area: str
) -> tuple[str | None, str | None]:
    configured_path = data_config.get("curriculum_context_path")
    if not configured_path:
        return None, None
    key = f"{subject}|{content_area}"
    context = load_curriculum_contexts(str(configured_path)).get(key)
    return context, key if context is not None else None


def build_splits(
    data_config: dict[str, Any],
    *,
    token_length: Callable[[list[dict[str, str]]], int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    examples: dict[str, TrainingExample] = {}
    filtered: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    max_prompt_tokens = data_config.get("max_prompt_tokens")

    for source in data_config["sources"]:
        for row in load_raw_rows(source):
            example, reason = normalize_row(row, data_config)
            if reason:
                filtered[reason] += 1
                continue
            assert example is not None
            if token_length is not None and max_prompt_tokens is not None:
                if token_length(example.prompt) > int(max_prompt_tokens):
                    filtered["prompt_too_long"] += 1
                    continue
            if example.item_id in examples:
                filtered["duplicate_item_id"] += 1
                continue
            examples[example.item_id] = example
            source_counts[example.source_kind] += 1

    seed = int(data_config.get("seed", 42))
    eval_fraction = float(data_config.get("eval_fraction", 0.1))
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("data.eval_fraction must be between 0 and 1")

    ranked = sorted(examples.values(), key=lambda item: _rank(item.item_id, seed))
    eval_count = max(1, round(len(ranked) * eval_fraction)) if ranked else 0
    eval_examples = ranked[:eval_count]
    train_examples = ranked[eval_count:]

    max_train = data_config.get("max_train_samples")
    max_eval = data_config.get("max_eval_samples")
    if max_train is not None:
        train_examples = train_examples[: int(max_train)]
    if max_eval is not None:
        eval_examples = eval_examples[: int(max_eval)]
    if not train_examples:
        raise ValueError("no training examples remain after filtering")
    if not eval_examples:
        raise ValueError("no evaluation examples remain after filtering")

    stats = {
        "eligible_unique": len(examples),
        "train_samples": len(train_examples),
        "eval_samples": len(eval_examples),
        "filtered": dict(sorted(filtered.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "curriculum_context_counts": dict(
            sorted(Counter(item.curriculum_context_key for item in ranked).items())
        ),
        "score_counts": dict(sorted(Counter(str(item.score) for item in ranked).items())),
        "train_item_ids": [item.item_id for item in train_examples],
        "eval_item_ids": [item.item_id for item in eval_examples],
    }
    return (
        [item.to_record() for item in train_examples],
        [item.to_record() for item in eval_examples],
        stats,
    )


def _rank(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def _content_id(row: dict[str, Any]) -> str:
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"generated_{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _answer_candidates(answer: str, problem: dict[str, Any]) -> list[str]:
    candidates = [answer]
    choices = problem.get("choices") if isinstance(problem.get("choices"), list) else []
    index = CIRCLED_CHOICE_INDEX.get(answer)
    if index is None and answer.isdigit() and 1 <= int(answer) <= 5:
        index = int(answer) - 1
    if index is not None and index < len(choices) and isinstance(choices[index], dict):
        content = str(choices[index].get("content") or "").strip()
        if content and content not in candidates:
            candidates.append(content)
    return candidates
