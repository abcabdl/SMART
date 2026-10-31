import ast
import re


def _normalize_body_indentation(lines):
    block_openers = re.compile(
        r"^(?:if|elif|else|for|while|try|except|finally|with|def|class|match|case)\b.*:\s*$"
    )
    normalized = []
    indent_stack = []
    for raw in lines:
        if not raw.strip():
            normalized.append("")
            continue
        stripped = raw.lstrip()
        lead = len(raw) - len(stripped)
        while indent_stack and lead < indent_stack[-1]:
            indent_stack.pop()
        max_allowed = (indent_stack[-1] + 4) if indent_stack else 0
        if lead > max_allowed:
            lead = max_allowed
        rebuilt = (" " * lead) + stripped
        normalized.append(rebuilt)
        if block_openers.match(stripped):
            indent_stack.append(lead + 4)
    return normalized


def _normalize_completion(text, entry_point):
    if text is None:
        return ""
    text = str(text).replace("\\n", "\n")
    if not entry_point:
        return text
    lines = text.lstrip("\n").splitlines()
    if not lines:
        return text
    candidate_module = "\n".join(lines)
    try:
        tree = ast.parse(candidate_module)
        top_level_defs = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if entry_point in top_level_defs:
            return candidate_module
    except Exception:
        pass

    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    positive_indents = [i for i in indents if i > 0]
    min_pos_indent = min(positive_indents) if positive_indents else 0
    if min_pos_indent > 0:
        new_lines = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            leading = len(line) - len(line.lstrip())
            trim = min(leading, min_pos_indent)
            new_lines.append(line[trim:])
        lines = new_lines

    lines = _normalize_body_indentation(lines)
    lines = [("    " + line) if line.strip() else line for line in lines]
    return "\n".join(lines)


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    entry_point = payload.get("entry_point")
    updated = _normalize_completion(text, entry_point)
    return {"completion": updated, "changed": updated != text}

