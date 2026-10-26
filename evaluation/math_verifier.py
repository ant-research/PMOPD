#!/usr/bin/env python3
"""Single strict MATH verifier shared by pilot, RL reward, and final evaluation."""
from __future__ import annotations
import re
from typing import Any, Dict, Optional

_BOX_START = re.compile(r"\\(?:boxed|fbox)\s*\{")


def extract_last_balanced_box(text: str) -> Optional[str]:
    """Return the content of the last balanced \boxed/\fbox, or None."""
    matches = list(_BOX_START.finditer(text or ""))
    if not matches:
        return None
    m = matches[-1]
    depth = 1
    for i in range(m.end(), len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[m.end():i].strip()
    return None


def has_balanced_boxed_answer(text: str) -> bool:
    value = extract_last_balanced_box(text)
    return value is not None and bool(value.strip())


def _clean_response(text: str) -> str:
    # Some models put a harmless label inside the required box.
    return re.sub(r"(?i)(\\(?:boxed|fbox)\s*\{\s*)ANSWER\s*:\s*", r"\1", text or "")


def math_verify_detail(response: str, answer: str) -> Dict[str, Any]:
    """Strict symbolic/LaTeX equivalence. Deliberately has no numeric fallback."""
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig
    cleaned = _clean_response(response)
    # Enforce the locked output protocol and avoid implicit last-number extraction.
    boxed_value = extract_last_balanced_box(cleaned)
    pred_cfg = [ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)]
    gold_cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
    result: Dict[str, Any] = {
        "correct": 0,
        "boxed": has_balanced_boxed_answer(cleaned),
        "pred_parsed": None,
        "gold_parsed": None,
        "error": None,
    }
    if boxed_value is None:
        return result
    # math_verify supports boxed; normalize fbox to boxed for identical semantics.
    parse_text = r"\boxed{" + boxed_value + "}"
    try:
        pred = parse(parse_text, extraction_config=pred_cfg,
                     fallback_mode="first_match", extraction_mode="first_match")
        gold = parse(f"${answer}$", extraction_config=gold_cfg,
                     fallback_mode="first_match", extraction_mode="first_match")
        result["pred_parsed"] = repr(pred)
        result["gold_parsed"] = repr(gold)
        result["correct"] = int(bool(pred and gold and verify(gold, pred)))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def math_verify_score(response: str, answer: str) -> int:
    return int(math_verify_detail(response, answer)["correct"])
