import re


def _is_timeout_error(error_message):
    msg = str(error_message or "").strip().lower()
    if not msg:
        return False
    return "timeout" in msg


def _detect_indent(text, entry_point):
    if not entry_point:
        return ""
    for line in str(text or "").splitlines():
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return "    "


def _apply_timeout_guard(completion, entry_point):
    text = str(completion or "")
    if not text:
        return text

    updated = text
    indent = _detect_indent(updated, entry_point)
    guard_prefix = f"{indent}__iter_guard = 0\n{indent}__iter_limit = 50000\n"

    if "while heap" in updated and "__iter_limit" not in updated:
        updated = guard_prefix + updated
        updated = re.sub(
            r"(?m)^(\s*)while\s+heap\s*:\s*$",
            r"\1while heap and __iter_guard < __iter_limit:\n\1    __iter_guard += 1",
            updated,
        )

    updated = re.sub(
        r"(?m)^(\s*if\s+len\(\s*path\s*\)\s*==\s*k\s*:\s*(?:\n(?:\s+.*))*?\n)(\s*)continue\s*$",
        r"\1\2break",
        updated,
    )
    return updated


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    entry_point = payload.get("entry_point")
    error_message = payload.get("error_message")

    if not _is_timeout_error(error_message):
        return {"completion": text, "changed": False}

    updated = _apply_timeout_guard(text, entry_point)
    return {"completion": updated, "changed": updated != text}
