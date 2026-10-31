import re


def _is_fallback(text):
    if text is None:
        return True
    raw = str(text)
    if not raw.strip():
        return True

    normalized = raw.strip().lower()
    if normalized in ("none", "null", "return none", "return null"):
        return True
    if "error calling llm" in normalized:
        return True

    lines = [line.strip().lower() for line in raw.splitlines() if line.strip()]
    if len(lines) == 1 and lines[0] in ("none", "null", "return none", "return null"):
        return True

    refusal_prefix = (
        "i'm sorry",
        "i need more",
        "could you please",
        "please provide",
        "more specific",
        "more information",
        "i apologize",
        "i cannot",
    )
    for prefix in refusal_prefix:
        if normalized.startswith(prefix):
            return True

    if ("sys.stdin.read" in raw or "input(" in raw) and ("import sys" in raw and "sys.stdin" in raw):
        return True

    if "def is_sorted_recursive" in raw or "def helper" in raw or "def check_" in raw:
        return True

    return False


def run(inputs):
    text = str((inputs or {}).get("completion") or "")
    fallback = _is_fallback(text)
    return {
        "completion": "" if fallback else text,
        "changed": bool(fallback),
        "is_fallback": bool(fallback),
    }

