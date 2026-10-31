import re

DATASET = 'math'
FOCUS = 'Work Time Motion'
PRIMARY_ROLE = 'planner'
KEYWORDS = ["work", "time", "motion", "rate", "math", "reasoning", "numeric"]


def run(inputs):
    question = str(inputs.get("question") or inputs.get("problem") or inputs.get("query") or inputs.get("task") or "")
    lower = question.lower()
    cue_hits = [word for word in KEYWORDS if word.lower() in lower]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", question)
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "cue_hits": cue_hits, "match_score": len(cue_hits), "number_count": len(numbers)}}
