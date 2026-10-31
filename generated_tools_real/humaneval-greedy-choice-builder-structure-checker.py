import ast
import re
import subprocess
import sys
import tempfile
import textwrap

DATASET = 'humaneval'
FOCUS = 'Greedy Choice'
PRIMARY_ROLE = 'builder'
TOOL_KIND = 'structure-checker'
KEYWORDS = ["greedy", "choice", "local", "optimize", "python", "function", "code"]
_DEF_RE = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*:", re.DOTALL)


def _extract_parameters(signature_text):
    params = []
    for item in str(signature_text or "").split(","):
        name = item.strip()
        if not name:
            continue
        name = name.split("=", 1)[0].strip()
        if not name:
            continue
        params.append(name)
    return params


def _strip_fences(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def _extract_signature_from_prompt(prompt, entry_point):
    prompt_text = str(prompt or "")
    match = _DEF_RE.search(prompt_text)
    resolved_entry = str(entry_point or "").strip()
    params = []
    signature_text = ""
    if match:
        resolved_entry = resolved_entry or match.group(1)
        params = _extract_parameters(match.group(2))
        signature_text = match.group(0)
    return resolved_entry, params, signature_text


def _inspect_functions(code_text):
    functions = {}
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return functions, None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            if getattr(node.args, "vararg", None) is not None:
                params.append("*" + str(node.args.vararg.arg))
            if getattr(node.args, "kwarg", None) is not None:
                params.append("**" + str(node.args.kwarg.arg))
            functions[node.name] = params
    return functions, tree


def _resolve_signature(inputs):
    prompt = str(inputs.get("prompt") or inputs.get("query") or inputs.get("task") or "")
    entry_point = str(inputs.get("entry_point") or "").strip()
    param_names = []
    if isinstance(inputs.get("param_names"), list):
        param_names = [str(item).strip() for item in inputs.get("param_names") if str(item).strip()]
    func_sig = inputs.get("function_signature")
    signature_text = ""
    if isinstance(func_sig, dict):
        if not param_names:
            param_names = [str(item).strip() for item in (func_sig.get("parameters") or []) if str(item).strip()]
        signature_text = str(func_sig.get("signature_text") or func_sig.get("signature") or "")
    prompt_entry, prompt_params, prompt_signature = _extract_signature_from_prompt(prompt, entry_point)
    entry_point = entry_point or prompt_entry
    if not param_names:
        param_names = prompt_params
    if not signature_text:
        signature_text = prompt_signature
    return prompt, entry_point, param_names, signature_text


def _indent_block(body):
    lines = str(body or "").splitlines()
    if not lines:
        return "    pass"
    return "\n".join(("    " + line) if line.strip() else "    " for line in lines)


def _canonicalize_code(inputs):
    raw_code = (
        inputs.get("code")
        or inputs.get("draft_code")
        or inputs.get("normalized_code")
        or inputs.get("query")
        or ""
    )
    raw_candidate = str(raw_code or "")
    if not raw_candidate.strip():
        raise Exception("Missing code")
    prompt, entry_point, param_names, signature_text = _resolve_signature(inputs)
    cleaned = textwrap.dedent(_strip_fences(raw_candidate)).strip("\n")
    functions, _ = _inspect_functions(cleaned)
    if functions and ((entry_point and entry_point in functions) or (not entry_point and functions)):
        if not entry_point:
            entry_point = next(iter(functions.keys()))
        if not param_names:
            param_names = list(functions.get(entry_point, []))
        wrapped_code = cleaned.rstrip() + "\n"
        code_shape = "raw_full_def"
    else:
        if not entry_point:
            raise Exception("Missing entry_point for body-only code")
        signature = "def %s(%s):" % (entry_point, ", ".join(param_names))
        wrapped_code = signature + "\n" + _indent_block(cleaned or "pass") + "\n"
        code_shape = "wrapped_body"
    return {
        "raw_candidate": raw_candidate,
        "cleaned_candidate": cleaned,
        "wrapped_code": wrapped_code,
        "code_shape": code_shape,
        "entry_point": entry_point or None,
        "param_names": list(param_names),
        "signature_text": signature_text,
        "prompt": prompt,
    }


def _split_test_lines(test, test_list):
    raw_lines = []
    if isinstance(test_list, list) and test_list:
        for item in test_list:
            raw_lines.extend(str(item).splitlines())
    else:
        raw_lines = str(test or "").splitlines()
    setup = []
    asserts = []
    for line in raw_lines:
        stripped = str(line).strip()
        if stripped.startswith("assert "):
            asserts.append(stripped)
        else:
            setup.append(str(line))
    return setup, asserts


def _rewrite_test_body(test, test_list, entry_point):
    body = "\n".join(str(item) for item in (test_list or [])) if test_list else str(test or "")
    if entry_point:
        body = body.replace("candidate(", "%s(" % entry_point)
        body = body.replace("check(candidate)", "check(%s)" % entry_point)
        body = "candidate = %s\n" % entry_point + body
    return body


def _run_python(program, timeout_s):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(program)
        path = fh.name
    completed = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout_s, check=False)
    return completed


def run(inputs):
    canonical = _canonicalize_code(inputs)
    try:
        tree = ast.parse(canonical["wrapped_code"])
    except SyntaxError as exc:
        return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "entry_point": canonical["entry_point"] or None, "function_found": False, "parameters": canonical["param_names"], "param_names": canonical["param_names"], "functions": [], "loop_count": 0, "branch_count": 0, "recursive": False, "normalized_code": canonical["wrapped_code"], "code_shape": canonical["code_shape"], "syntax_error": str(exc)}}
    functions = {}
    loops = 0
    branches = 0
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            if getattr(node.args, "vararg", None) is not None:
                params.append("*" + str(node.args.vararg.arg))
            if getattr(node.args, "kwarg", None) is not None:
                params.append("**" + str(node.args.kwarg.arg))
            functions[node.name] = params
        elif isinstance(node, (ast.For, ast.While)):
            loops += 1
        elif isinstance(node, ast.If):
            branches += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
    entry_point = canonical["entry_point"]
    found = entry_point in functions if entry_point else False
    recursive = entry_point in calls if entry_point else False
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "entry_point": entry_point or None, "function_found": found, "parameters": functions.get(entry_point, canonical["param_names"]), "param_names": functions.get(entry_point, canonical["param_names"]), "functions": list(functions.keys()), "loop_count": loops, "branch_count": branches, "recursive": recursive, "normalized_code": canonical["wrapped_code"], "code_shape": canonical["code_shape"]}}
