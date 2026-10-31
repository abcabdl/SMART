import ast
import re
import textwrap


def _extract_code_block(text):
    if not text:
        return text
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) < 3:
        return text
    code = parts[1]
    lines = code.splitlines()
    if lines and lines[0].strip().lower() in {"python", "py", "json"}:
        lines = lines[1:]
    extracted = "\n".join(lines).strip()
    return extracted if extracted else text


def _strip_redundant_def(text, entry_point):
    if not text or not entry_point:
        return text
    pattern = re.compile(rf"^\s*def\s+{re.escape(str(entry_point))}\s*\(")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if pattern.match(line):
            body = lines[idx + 1 :]
            if body:
                return "\n".join(body).lstrip("\n")
            return text
    return text


def _repair_completion(completion, entry_point):
    text = str(completion or "")
    text = text.replace("\r\n", "\n")
    text = _extract_code_block(text)
    text = text.expandtabs(4)
    text = _strip_redundant_def(text, entry_point)
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    text = "\n".join(lines)
    text = textwrap.dedent(text).strip("\n")
    text = _repair_missing_block_indentation(text)
    if entry_point:
        body_lines = [("    " + line) if line.strip() else line for line in text.splitlines()]
        text = "\n".join(body_lines)
    return text


def _repair_missing_block_indentation(text):
    lines = str(text or "").splitlines()
    if not lines:
        return str(text or "")

    block_openers = re.compile(
        r"^(?:if|elif|else|for|while|try|except|finally|with|def|class|match|case)\b.*:\s*$"
    )
    same_level_after_block = ("elif", "else", "except", "finally")

    repaired = []
    expected_indent = None
    for raw in lines:
        stripped = raw.lstrip()
        if not stripped:
            repaired.append("")
            continue

        lead = len(raw) - len(stripped)
        if expected_indent is not None:
            is_same_level = any(stripped.startswith(prefix) for prefix in same_level_after_block)
            if not is_same_level and lead < expected_indent:
                lead = expected_indent
        fixed = (" " * lead) + stripped
        repaired.append(fixed)

        if block_openers.match(stripped):
            expected_indent = lead + 4
        else:
            expected_indent = None

    return "\n".join(repaired)


def _has_body_syntax_error(text, param_names):
    params = ", ".join(param_names) if param_names else "a, b"
    wrapped = f"def __candidate({params}):\n"
    lines = text.splitlines() or ["pass"]
    for line in lines:
        wrapped += ("    " + line if line.strip() else "    ") + "\n"
    try:
        ast.parse(wrapped)
    except Exception:
        return True
    return False


def _normalize_param_names(param_names):
    if not isinstance(param_names, list):
        return []
    cleaned = []
    for item in param_names:
        value = str(item).strip()
        if value:
            cleaned.append(value)
    return cleaned


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    entry_point = payload.get("entry_point")
    param_names = _normalize_param_names(payload.get("param_names"))
    enabled = bool(payload.get("enable_syntax_repair", True))

    if not enabled:
        return {"completion": text, "changed": False}
    if not _has_body_syntax_error(text, param_names):
        return {"completion": text, "changed": False}

    repaired = _repair_completion(text, entry_point)
    if not repaired:
        return {"completion": text, "changed": False}
    if _has_body_syntax_error(repaired, param_names):
        return {"completion": text, "changed": False}
    return {"completion": repaired, "changed": repaired != text}
