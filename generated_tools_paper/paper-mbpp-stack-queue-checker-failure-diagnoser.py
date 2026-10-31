DATASET = 'mbpp'
FOCUS = 'Stack Queue'
PRIMARY_ROLE = 'checker'
KEYWORDS = ["stack", "queue", "push", "pop", "python", "function", "code"]


def run(inputs):
    code = str(inputs.get("code") or "")
    error = str(inputs.get("traceback") or inputs.get("error") or "")
    lines = code.splitlines()
    focus_lines = []
    for raw in error.splitlines():
        if ", line " not in raw:
            continue
        try:
            line_no = int(raw.split(", line ", 1)[1].split(",", 1)[0].strip())
        except Exception:
            continue
        context = []
        for pos in range(max(0, line_no - 3), min(len(lines), line_no + 2)):
            context.append({"line": pos + 1, "code": lines[pos]})
        focus_lines.append({"line": line_no, "context": context})
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "error_summary": error.splitlines()[-1] if error.splitlines() else error, "focus_lines": focus_lines}}
