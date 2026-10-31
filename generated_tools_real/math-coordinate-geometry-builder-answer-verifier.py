import re

DATASET = 'math'
FOCUS = 'Coordinate Geometry'
PRIMARY_ROLE = 'builder'
KEYWORDS = ["coordinate", "distance", "midpoint", "slope", "math", "reasoning", "numeric"]
NUM_RE = re.compile(r"[-+]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))(?:/[-+]?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+)))?")


def _parse_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.startswith("####"):
        text = text[4:].strip()
    nums = NUM_RE.findall(text)
    if not nums:
        raise Exception("No numeric token found")
    token = nums[-1]
    if "/" in token:
        left, right = token.split("/", 1)
        return float(left) / float(right)
    return float(token)


def run(inputs):
    candidate = inputs.get("candidate")
    expected = inputs.get("expected")
    if candidate is None:
        raise Exception("Missing candidate")
    candidate_value = _parse_value(candidate)
    result = {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "candidate_value": candidate_value}
    if expected is not None:
        expected_value = _parse_value(expected)
        result["expected_value"] = expected_value
        result["matches_expected"] = abs(candidate_value - expected_value) <= float(inputs.get("tolerance") or 1e-6)
    return {"output": result}
