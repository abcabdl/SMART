import ast
import difflib
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


def _is_name_resolution_error(error_message):
    text = str(error_message or "")
    return ("NameError" in text) or ("UnboundLocalError" in text)


def _extract_missing_symbol(error_message):
    msg = str(error_message or "")
    if not msg:
        return ""
    match = re.search(r"NameError:\s+name '([A-Za-z_][A-Za-z0-9_]*)' is not defined", msg)
    if match:
        return match.group(1)
    match = re.search(
        r"UnboundLocalError:\s+cannot access local variable '([A-Za-z_][A-Za-z0-9_]*)'",
        msg,
    )
    if match:
        return match.group(1)
    return ""


def _extract_suggested_symbol(error_message):
    msg = str(error_message or "")
    if not msg:
        return ""
    match = re.search(r"Did you mean:\s*'([A-Za-z_][A-Za-z0-9_]*)'\?", msg)
    if match:
        return match.group(1)
    return ""


def _normalize_identifier(name):
    return re.sub(r"_+", "", str(name or "").strip().lower())


def _relation_score(missing, param_name):
    m_raw = str(missing or "").strip()
    p_raw = str(param_name or "").strip()
    if not m_raw or not p_raw:
        return 0.0

    m = _normalize_identifier(m_raw)
    p = _normalize_identifier(p_raw)
    if not m or not p:
        return 0.0

    if m == p:
        return 1.0
    if p.startswith(m) and len(m) >= 3:
        return 0.95
    if m.startswith(p) and len(p) >= 2:
        return 0.90

    alias_groups = [
        {"n", "num", "number", "val", "value", "x", "target"},
        {"s", "str", "string", "text", "txt", "sentence", "word", "words", "chars"},
        {"arr", "array", "list", "lst", "nums", "values", "items", "seq", "sequence", "grid", "data"},
        {"i", "idx", "index", "pos", "pointer"},
        {"a", "b", "left", "right", "first", "second"},
    ]
    for group in alias_groups:
        if m in group and p in group:
            return 0.90

    return difflib.SequenceMatcher(a=m, b=p).ratio()


def _select_target_param(missing, param_names, suggested=""):
    if not missing or not param_names:
        return ""
    if suggested and suggested in param_names and suggested != missing:
        return suggested
    scored = sorted(
        ((name, _relation_score(missing, name)) for name in param_names),
        key=lambda item: item[1],
        reverse=True,
    )
    best_name, best_score = scored[0]
    if best_score < 0.86:
        return ""
    if len(scored) > 1 and scored[1][1] >= 0.86 and (best_score - scored[1][1]) < 0.04:
        return ""
    return best_name


def _line_offsets(text):
    offsets = [0]
    total = 0
    for line in text.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _to_offset(offsets, lineno, col):
    return offsets[lineno - 1] + col


def _collect_load_name_spans(scope_node, missing):
    spans = []

    def _walk(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == missing:
            end_lineno = getattr(node, "end_lineno", node.lineno)
            end_col = getattr(node, "end_col_offset", node.col_offset + len(missing))
            spans.append((node.lineno, node.col_offset, end_lineno, end_col))
            return
        for child in ast.iter_child_nodes(node):
            _walk(child)

    for stmt in getattr(scope_node, "body", []):
        _walk(stmt)
    return spans


def _apply_spans(text, spans, replacement):
    if not spans:
        return text
    offsets = _line_offsets(text)
    ranges = []
    for lineno, col, end_lineno, end_col in spans:
        start = _to_offset(offsets, lineno, col)
        end = _to_offset(offsets, end_lineno, end_col)
        ranges.append((start, end))
    ranges.sort(key=lambda item: item[0], reverse=True)
    updated = text
    for start, end in ranges:
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _regex_replace_identifier(text, missing, target):
    if not text or not missing or not target or missing == target:
        return text
    pattern = re.compile(rf"\b{re.escape(missing)}\b")
    return pattern.sub(target, text)


def _find_entry_function(tree, entry_point):
    if not entry_point:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return node
    return None


def _rewrite_module_text(text, entry_point, missing, target):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False

    fn = _find_entry_function(tree, entry_point)
    if fn is None:
        return text, False

    spans = _collect_load_name_spans(fn, missing)
    if not spans:
        return text, False
    updated = _apply_spans(text, spans, target)
    return updated, updated != text


def _wrap_body(text, param_names):
    params = ", ".join(param_names) if param_names else "_unused"
    lines = str(text or "").splitlines()
    wrapped = [f"def __candidate({params}):"]
    if not lines:
        wrapped.append("    pass")
        return "\n".join(wrapped)
    for line in lines:
        if line.strip():
            wrapped.append("    " + line)
        else:
            wrapped.append("    ")
    return "\n".join(wrapped)


def _unwrap_body(wrapped_text):
    lines = str(wrapped_text or "").splitlines()
    body = lines[1:] if len(lines) > 1 else []
    unwrapped = []
    for line in body:
        if line.startswith("    "):
            unwrapped.append(line[4:])
        elif not line.strip():
            unwrapped.append("")
        else:
            unwrapped.append(line)
    return "\n".join(unwrapped)


def _rewrite_body_text(text, param_names, missing, target):
    wrapped = _wrap_body(text, param_names)
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return text, False
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return text, False
    fn = tree.body[0]
    spans = _collect_load_name_spans(fn, missing)
    if not spans:
        return text, False
    wrapped_updated = _apply_spans(wrapped, spans, target)
    updated = _unwrap_body(wrapped_updated)
    return updated, updated != text


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    params = _normalize_param_names(payload.get("param_names"))
    error_message = payload.get("error_message")
    entry_point = payload.get("entry_point")

    if not text or not params or not _is_name_resolution_error(error_message):
        return {"completion": text, "changed": False}

    missing = _extract_missing_symbol(error_message)
    if not missing or missing in params:
        return {"completion": text, "changed": False}

    suggested = _extract_suggested_symbol(error_message)
    target = _select_target_param(missing, params, suggested=suggested)
    if not target or target == missing:
        return {"completion": text, "changed": False}

    updated, changed = _rewrite_module_text(text, entry_point, missing, target)
    if not changed:
        updated, changed = _rewrite_body_text(text, params, missing, target)
    if not changed:
        fallback = _regex_replace_identifier(text, missing, target)
        changed = fallback != text
        updated = fallback
    return {"completion": updated, "changed": changed}
