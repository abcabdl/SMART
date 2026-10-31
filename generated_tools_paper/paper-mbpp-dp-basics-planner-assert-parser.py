import ast

DATASET = 'mbpp'
FOCUS = 'DP Basics'
PRIMARY_ROLE = 'planner'
KEYWORDS = ["dp", "tabulation", "memoization", "state", "python", "function", "code"]


def run(inputs):
    prompt = str(inputs.get("prompt") or "")
    test = str(inputs.get("test") or "")
    test_list = [str(item) for item in (inputs.get("test_list") or [])]
    doctests = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith(">>>"):
            doctests.append(line[3:].strip())
    asserts = []
    for raw in test_list or test.splitlines():
        line = str(raw).strip()
        if not line.startswith("assert "):
            continue
        try:
            node = ast.parse(line).body[0]
            if isinstance(node, ast.Assert):
                asserts.append(ast.unparse(node.test) if hasattr(ast, "unparse") else line)
        except SyntaxError:
            asserts.append(line)
    cue_hits = [word for word in KEYWORDS if word.lower() in (prompt + "\n" + test).lower()]
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "doctests": doctests[:20], "asserts": asserts[:20], "cue_hits": cue_hits, "example_count": len(doctests) + len(asserts)}}
