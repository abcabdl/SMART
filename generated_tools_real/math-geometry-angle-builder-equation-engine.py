import math
import re

DATASET = 'math'
FOCUS = 'Geometry Angle'
PRIMARY_ROLE = 'builder'
KEYWORDS = ["angle", "triangle", "degree", "trisector", "math", "reasoning", "numeric"]


def _parse_polynomial_side(text, variable):
    coeffs = {}
    normalized = text.replace("-", "+-").replace(" ", "")
    for term in [item for item in normalized.split("+") if item]:
        if variable not in term:
            coeffs[0] = coeffs.get(0, 0.0) + float(term)
            continue
        if term == variable:
            coeffs[1] = coeffs.get(1, 0.0) + 1.0
            continue
        if term == "-" + variable:
            coeffs[1] = coeffs.get(1, 0.0) - 1.0
            continue
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\*?" + re.escape(variable) + r"(?:\*\*(\d+))?", term)
        if not match:
            raise ValueError("Unsupported term")
        coefficient = float(match.group(1))
        power = int(match.group(2) or "1")
        coeffs[power] = coeffs.get(power, 0.0) + coefficient
    return coeffs


def _combine(lhs, rhs):
    powers = set(lhs) | set(rhs)
    return {power: lhs.get(power, 0.0) - rhs.get(power, 0.0) for power in powers}


def _solve_linear(coeffs):
    a = coeffs.get(1, 0.0)
    b = coeffs.get(0, 0.0)
    if abs(a) <= 1e-12:
        raise ValueError("Not a linear equation")
    return [(-b) / a]


def _solve_quadratic(coeffs):
    a = coeffs.get(2, 0.0)
    b = coeffs.get(1, 0.0)
    c = coeffs.get(0, 0.0)
    if abs(a) <= 1e-12:
        return _solve_linear(coeffs)
    discriminant = b * b - 4.0 * a * c
    if discriminant < -1e-12:
        return []
    discriminant = max(discriminant, 0.0)
    root = math.sqrt(discriminant)
    return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]


def run(inputs):
    equation = str(inputs.get("equation") or "").strip().replace("^", "**")
    variable = str(inputs.get("variable") or "x").strip() or "x"
    if not equation or "=" not in equation:
        raise Exception("Missing equation")
    left, right = equation.split("=", 1)
    coeffs = _combine(_parse_polynomial_side(left, variable), _parse_polynomial_side(right, variable))
    degree = max(coeffs) if coeffs else 0
    if degree <= 1:
        solutions = _solve_linear(coeffs)
    elif degree == 2:
        solutions = _solve_quadratic(coeffs)
    else:
        raise ValueError("Only linear and quadratic equations are supported")
    normalized = []
    seen = set()
    for item in solutions:
        rounded = round(float(item), 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        normalized.append(rounded)
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "equation": equation, "variable": variable, "solutions": normalized, "solution_count": len(normalized)}}
