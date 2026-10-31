import json
import re


def _unwrap_json_code(text):
    if not text or "{" not in text:
        return text
    start = text.find("{")
    if start < 0:
        return text
    try:
        data = json.loads(text[start:])
    except Exception:
        return text
    if not isinstance(data, dict):
        return text
    for key in ("code_or_commands", "code", "solution", "output"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            for nested in ("code_or_commands", "code", "solution"):
                nested_value = value.get(nested)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value
    return text


def _extract_code_block(text):
    if not text:
        return text
    text = _unwrap_json_code(text)
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


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    entry_point = payload.get("entry_point")
    should_apply = (
        ("```" in text)
        or ("\\n" in text)
        or ("\\t" in text)
        or (bool(entry_point) and bool(re.search(rf"^\s*def\s+{re.escape(str(entry_point))}\s*\(", text, re.M)))
    )
    if not should_apply:
        return {"completion": text, "changed": False}

    updated = text.replace("\r\n", "\n")
    updated = updated.replace("\\n", "\n").replace("\\t", "\t")
    updated = _extract_code_block(updated)
    updated = _strip_redundant_def(updated, entry_point)
    return {"completion": updated, "changed": updated != text}

