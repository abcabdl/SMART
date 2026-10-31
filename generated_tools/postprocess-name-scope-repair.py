import ast
import difflib
import re


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


def _is_name_resolution_error(error_message):
    msg = str(error_message or "")
    return ("NameError" in msg) or ("UnboundLocalError" in msg)


def _normalize_param_names(param_names):
    if not isinstance(param_names, list):
        return []
    normalized = []
    for item in param_names:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_identifier(name):
    return re.sub(r"_+", "", str(name or "").strip().lower())


def _relation_score(missing, candidate):
    m_raw = str(missing or "").strip()
    c_raw = str(candidate or "").strip()
    if not m_raw or not c_raw:
        return 0.0

    m = _normalize_identifier(m_raw)
    c = _normalize_identifier(c_raw)
    if not m or not c:
        return 0.0

    if m == c:
        return 1.0
    if c.startswith(m) and len(m) >= 3:
        return 0.95
    if m.startswith(c) and len(c) >= 2:
        return 0.90

    alias_groups = [
        {"n", "num", "number", "val", "value", "x", "target"},
        {"s", "str", "string", "text", "txt", "sentence", "word", "words", "chars"},
        {"arr", "array", "list", "lst", "nums", "values", "items", "seq", "sequence", "grid", "data"},
        {"i", "idx", "index", "pos", "pointer"},
        {"a", "b", "left", "right", "first", "second"},
    ]
    for group in alias_groups:
        if m in group and c in group:
            return 0.90
    return difflib.SequenceMatcher(a=m, b=c).ratio()


def _collect_identifier_candidates(text):
    if not text:
        return []
    names = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
    return sorted(set(names))


def _select_symbol_replacement(missing, suggested, param_names, text):
    if not missing:
        return ""
    if suggested and suggested != missing:
        return suggested

    candidates = list(param_names or [])
    for name in _collect_identifier_candidates(text):
        if name not in candidates:
            candidates.append(name)
    candidates = [name for name in candidates if name and name != missing]
    if not candidates:
        return ""

    scored = sorted(
        ((name, _relation_score(missing, name)) for name in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    best_name, best_score = scored[0]
    if best_score < 0.92:
        return ""
    if len(scored) > 1 and scored[1][1] >= 0.92 and (best_score - scored[1][1]) < 0.04:
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

    scope = _find_entry_function(tree, entry_point)
    if scope is None:
        scope = tree

    spans = _collect_load_name_spans(scope, missing)
    if not spans:
        return text, False
    updated = _apply_spans(text, spans, target)
    return updated, updated != text


def _wrap_body(text):
    lines = str(text or "").splitlines()
    wrapped = ["def __candidate():"]
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


def _rewrite_body_text(text, missing, target):
    wrapped = _wrap_body(text)
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


def _detect_indent(text):
    for line in str(text or "").splitlines():
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return "    "


def _indent_block_preserve_relative(block, base_indent):
    raw_lines = str(block or "").strip("\n").splitlines()
    if not raw_lines:
        return ""
    non_empty = [line for line in raw_lines if line.strip()]
    if not non_empty:
        return ""

    min_leading = min(len(line) - len(line.lstrip()) for line in non_empty)
    indented = []
    for line in raw_lines:
        if not line.strip():
            indented.append("")
            continue
        leading = len(line) - len(line.lstrip())
        relative = max(leading - min_leading, 0)
        indented.append(base_indent + (" " * relative) + line.lstrip())
    return "\n".join(indented)


def _prepend_block(text, block, indent):
    formatted = _indent_block_preserve_relative(block, indent)
    if not formatted:
        return text
    stripped = str(text or "").lstrip("\n")
    if not stripped:
        return formatted
    return formatted + "\n" + stripped


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    entry_point = payload.get("entry_point")
    param_names = _normalize_param_names(payload.get("param_names"))
    error_message = payload.get("error_message")
    missing = _extract_missing_symbol(error_message)
    if not text or not missing or not _is_name_resolution_error(error_message):
        return {"completion": text, "changed": False}

    # Do not mutate logic-level assertion failures.
    if "AssertionError" in str(error_message or ""):
        return {"completion": text, "changed": False}

    suggested = _extract_suggested_symbol(error_message)
    target_symbol = _select_symbol_replacement(
        missing=missing,
        suggested=suggested,
        param_names=param_names,
        text=text,
    )

    updated = text
    if target_symbol:
        updated, changed = _rewrite_module_text(updated, entry_point, missing, target_symbol)
        if not changed:
            updated, changed = _rewrite_body_text(updated, missing, target_symbol)
        if changed:
            return {"completion": updated, "changed": True}

    indent = _detect_indent(updated)

    module_symbols = {"math", "heapq", "itertools", "re", "collections", "functools"}
    if missing in module_symbols and f"import {missing}" not in updated:
        updated = _prepend_block(updated, f"import {missing}", indent)

    symbol_imports = {
        "heappush": "from heapq import heappush, heappop",
        "heappop": "from heapq import heappush, heappop",
        "deque": "from collections import deque",
        "defaultdict": "from collections import defaultdict",
        "Counter": "from collections import Counter",
        "ceil": "from math import ceil",
        "sqrt": "from math import sqrt",
    }
    import_stmt = symbol_imports.get(missing)
    if import_stmt and import_stmt not in updated:
        updated = _prepend_block(updated, import_stmt, indent)

    helper_templates = {
        "is_prime": (
            "def is_prime(n):\n"
            "    if n <= 1:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True"
        ),
        "convert_to_float": (
            "def convert_to_float(value):\n"
            "    if isinstance(value, str):\n"
            "        value = value.replace(',', '.')\n"
            "    try:\n"
            "        return float(value)\n"
            "    except (ValueError, TypeError):\n"
            "        return None"
        ),
    }
    helper = helper_templates.get(missing)
    if helper and f"def {missing}" not in updated and f"{missing}(" in updated:
        updated = _prepend_block(updated, helper, indent)

    return {"completion": updated, "changed": updated != text}
