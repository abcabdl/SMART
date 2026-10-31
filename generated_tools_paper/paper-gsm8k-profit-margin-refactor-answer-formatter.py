import re

DATASET = 'gsm8k'
FOCUS = 'Profit Margin'
PRIMARY_ROLE = 'refactor'
KEYWORDS = ["profit", "cost", "revenue", "margin", "math", "reasoning", "numeric"]
BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def _canonical(text):
    value = str(text).strip()
    boxes = BOX_RE.findall(value)
    if boxes:
        value = boxes[-1].strip()
    if value.startswith("####"):
        value = value[4:].strip()
    return value.replace("$", "").strip()


def run(inputs):
    answer = inputs.get("answer")
    if answer is None:
        answer = inputs.get("candidate")
    if answer is None:
        answer = inputs.get("value")
    if answer is None:
        raise Exception("Missing answer")
    canonical = _canonical(answer)
    formatted = "#### " + canonical if DATASET == "gsm8k" else canonical
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "canonical_answer": canonical, "formatted_answer": formatted}}
