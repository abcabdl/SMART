import re


def _normalize_param_names(param_names):
    if not isinstance(param_names, list):
        return []
    normalized = []
    for item in param_names:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _repair_builtin_shadowing(completion, param_names):
    text = str(completion or "")
    if not text or not param_names:
        return text

    type_expr_by_name = {
        "str": "type('')",
        "int": "type(0)",
        "float": "type(0.0)",
        "bool": "type(True)",
        "list": "type([])",
        "dict": "type({})",
        "tuple": "type(())",
        "set": "type({1})",
        "bytes": "type(b'')",
    }

    for name in param_names:
        type_expr = type_expr_by_name.get(name)
        if not type_expr:
            continue
        text = re.sub(
            rf"\bisinstance\(\s*{re.escape(name)}\s*,\s*{re.escape(name)}\s*\)",
            f"isinstance({name}, {type_expr})",
            text,
        )
        text = re.sub(
            rf"\btype\(\s*{re.escape(name)}\s*\)\s+is\s+{re.escape(name)}\b",
            f"type({name}) is {type_expr}",
            text,
        )
        text = re.sub(
            rf"\btype\(\s*{re.escape(name)}\s*\)\s*==\s*{re.escape(name)}\b",
            f"type({name}) == {type_expr}",
            text,
        )
    return text


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    params = _normalize_param_names(payload.get("param_names"))
    if not params:
        return {"completion": text, "changed": False}
    updated = _repair_builtin_shadowing(text, params)
    return {"completion": updated, "changed": updated != text}

