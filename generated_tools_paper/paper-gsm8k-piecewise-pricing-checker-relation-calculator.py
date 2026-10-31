import ast
import math
import re

DATASET = 'gsm8k'
FOCUS = 'Piecewise Pricing'
PRIMARY_ROLE = 'checker'
KEYWORDS = ["pricing", "tiers", "bands", "cost", "math", "reasoning", "numeric"]


def _normalize(expr):
    text = str(expr or "").strip()
    text = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    return text.replace("^", "**")


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        funcs = {"sqrt": math.sqrt, "abs": abs, "round": round, "factorial": math.factorial, "comb": math.comb, "perm": math.perm}
        if node.func.id not in funcs:
            raise ValueError("Unsupported function")
        return funcs[node.func.id](*[_eval(arg) for arg in node.args])
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError("Unsupported symbol")
    raise ValueError("Unsupported expression")


def run(inputs):
    expression = inputs.get("expression")
    if expression is None:
        if all(key in inputs for key in ("total", "unit_price")):
            value = float(inputs["total"]) / float(inputs["unit_price"])
            return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "mode": "total_over_unit_price", "value": value}}
        if all(key in inputs for key in ("distance", "speed")):
            value = float(inputs["distance"]) / float(inputs["speed"])
            return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "mode": "distance_over_speed", "value": value}}
        raise Exception("Missing expression or structured numeric inputs")
    normalized = _normalize(expression)
    value = _eval(ast.parse(normalized, mode="eval"))
    return {"output": {"dataset": DATASET, "focus": FOCUS, "primary_role": PRIMARY_ROLE, "expression": expression, "normalized_expression": normalized, "value": value}}
