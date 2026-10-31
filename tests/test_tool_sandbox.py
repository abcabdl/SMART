import importlib
import sys
from pathlib import Path


def _import_from_src(module: str):
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    return importlib.import_module(module)


def test_tool_sandbox_execution(tmp_path) -> None:
    tool_gen_mod = _import_from_src("generation.tool_generator")
    sandbox_mod = _import_from_src("execution.sandbox")

    synth = tool_gen_mod.MockToolSynthesizer()
    code, _ = synth.synthesize("text_cleaner")
    result = sandbox_mod._execute_tool_code(code, {"text": " Hello, World! "})

    assert result["text"] == "hello world"


def test_tool_sandbox_exposes_syntaxerror_builtin() -> None:
    sandbox_mod = _import_from_src("execution.sandbox")

    tool_code = """
def run(inputs):
    try:
        compile("return 1", "<tool>", "exec")
    except SyntaxError:
        return {"output": {"caught": True}}
    return {"output": {"caught": False}}
"""

    result = sandbox_mod._execute_tool_code(tool_code, {})
    assert result["output"]["caught"] is True


def test_tool_sandbox_exposes_common_introspection_builtins() -> None:
    sandbox_mod = _import_from_src("execution.sandbox")

    tool_code = """
class Thing:
    value = 3

def run(inputs):
    thing = Thing()
    return {
        "output": {
            "has": hasattr(thing, "value"),
            "value": getattr(thing, "value"),
            "mapped": list(map(lambda x: x + 1, [1, 2])),
        }
    }
"""

    result = sandbox_mod._execute_tool_code(tool_code, {})
    assert result["output"] == {"has": True, "value": 3, "mapped": [2, 3]}
