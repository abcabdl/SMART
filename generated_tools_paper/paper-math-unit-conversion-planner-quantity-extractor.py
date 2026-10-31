import re

DATASET = 'math'
FOCUS = 'Unit Conversion'
PRIMARY_ROLE = 'planner'
KEYWORDS = ["unit", "convert", "measurement", "dozen", "math", "reasoning", "numeric"]
UNITS = ["dollar", "dollars", "percent", "%", "hour", "hours", "day", "days", "mile", "miles", "degree", "degrees", "dozen", "feet", "meters"]


def run(inputs):
    text = str(inputs.get("question") or inputs.get("problem") or inputs.get("query") or inputs.get("task") or "")
    lower = text.lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    fractions = re.findall(r"\b\d+\s*/\s*\d+\b", text)
    units = [unit for unit in UNITS if unit in lower]
    cue_hits = [word for word in KEYWORDS if word.lower() in lower]
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "numbers": numbers, "fractions": fractions, "units": units, "cue_hits": cue_hits}}
