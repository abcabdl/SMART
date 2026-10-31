import json
import argparse
import os
import sys
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure we can import from src
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation.llm_client import LLMClient


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _is_content_filter_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "content_filter",
        "content management policy",
        "response was filtered",
        "prompt triggering",
    ]
    return any(marker in text for marker in markers)


def _extract_prompt_features(prompt: str, dataset: str) -> Dict[str, Any]:
    lower = (prompt or "").lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", lower)
    latex_tokens = re.findall(
        r"\\(frac|sqrt|sum|prod|begin|end|cdot|equiv|pmod|theta|pi|angle|overline)",
        lower,
    )
    keyword_bank = [
        "triangle",
        "circle",
        "polygon",
        "matrix",
        "vector",
        "probability",
        "prime",
        "divisor",
        "gcd",
        "lcm",
        "perimeter",
        "area",
        "mod",
        "ratio",
        "percent",
        "sequence",
    ]
    tags = [kw for kw in keyword_bank if kw in lower]
    return {
        "dataset": dataset,
        "char_count": len(prompt or ""),
        "number_count": len(numbers),
        "sample_numbers": numbers[:8],
        "latex_token_count": len(latex_tokens),
        "has_asy": "[asy]" in lower,
        "has_table": "\\begin{tabular}" in lower,
        "keyword_tags": tags[:12],
    }


def _build_sanitized_task_text(problem: Dict[str, Any], dataset: str) -> str:
    features = _extract_prompt_features(str(problem.get("prompt") or ""), dataset)
    if dataset in {"gsm8k", "math"}:
        domain = "GSM8K word problem"
    elif dataset == "math":
        domain = "MATH competition problem"
    elif dataset == "mbpp":
        domain = "MBPP programming problem"
    else:
        domain = "HumanEval-style coding task"

    return (
        f"Task: Decide the best topology/roles for a {domain}.\n"
        "Safety mode is enabled: use ONLY metadata below, do NOT solve the original task, "
        "and return only the JSON object required by the system prompt.\n\n"
        "Problem metadata:\n"
        f"- dataset: {features['dataset']}\n"
        f"- char_count: {features['char_count']}\n"
        f"- number_count: {features['number_count']}\n"
        f"- sample_numbers: {features['sample_numbers']}\n"
        f"- latex_token_count: {features['latex_token_count']}\n"
        f"- has_asy: {features['has_asy']}\n"
        f"- has_table: {features['has_table']}\n"
        f"- keyword_tags: {features['keyword_tags']}\n"
    )


def _extract_first_function_from_code(code: str) -> Any:
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _extract_entry_point_from_code(code: str) -> str:
    fn = _extract_first_function_from_code(code)
    if fn is not None:
        return fn.name
    m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code or "")
    return m.group(1) if m else ""


def _extract_entry_point_from_tests(test_list: List[Any]) -> str:
    for test in test_list:
        m = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(test))
        if m:
            return m.group(1)
    return ""


def get_system_prompt(
    strategy: str = "single_preference",
    available_roles: Optional[List[str]] = None,
    dataset: str = "humaneval",
) -> str:
    """
    Generate system prompt based on agent topology strategy.
    
    Args:
        strategy: One of:
            - "single_preference" (default): Strongly encourages single agent (current behavior)
            - "balanced": Neutral stance, chooses based on task complexity
            - "multi_preference": Moderately encourages multi-agent for complex tasks
            - "multi_aggressive": Strongly encourages multi-agent collaboration
    """
    
    roles = [r.lower() for r in (available_roles or []) if str(r).strip()]
    if not roles:
        roles = ["builder", "planner", "checker", "refactor"]
    role_set = set(roles)
    roles_text = ", ".join(roles)

    role_policy_lines: List[str] = [
        "**Role Trigger Rules (query-driven, not fixed combo):**",
        "- Always choose the MINIMAL set of roles needed for this specific query.",
        "- Do not always return the same default role combination.",
        "- roles MUST be a subset of Available roles.",
    ]
    if dataset == "gsm8k":
        if "builder" in role_set:
            role_policy_lines.append(
                "- Use `builder` to execute primary arithmetic reasoning and produce the numeric answer."
            )
        if "planner" in role_set:
            role_policy_lines.append(
                "- Use `planner` when the word problem needs decomposition into equations/sub-steps."
            )
        if "checker" in role_set:
            role_policy_lines.append(
                "- Use `checker` for arithmetic validation, unit consistency checks, and final-number verification."
            )
        if "refactor" in role_set:
            role_policy_lines.append(
                "- Use `refactor` for second-pass cleanup of reasoning trace and strict final-output formatting."
            )
            role_policy_lines.append(
                "- Do NOT add `refactor` for very simple one-step arithmetic."
            )
        role_policy_lines.append(
            "- Prefer `single` with one role for direct one-step arithmetic; add roles only when multi-step verification is needed."
        )
        base_prompt = (
            "You are a meta-router for GSM8K math word problems. Decide the best topology.\n"
            "Return JSON: {\"topology\": \"...\", \"roles\": [...], \"manager_role\": \"...\", "
            "\"entry_role\": \"...\", \"max_steps\": N}\n\n"
            f"Available roles: {roles_text}\n"
        )
    else:
        if "builder" in role_set:
            role_policy_lines.append(
                "- Use `builder` when implementation/coding is required (usually required for code tasks)."
            )
        if "planner" in role_set:
            role_policy_lines.append(
                "- Use `planner` when task needs decomposition, algorithm choice, dependency ordering, or multi-step reasoning."
            )
        if "checker" in role_set:
            role_policy_lines.append(
                "- Use `checker` when correctness risk is high: edge cases, invariants, boundary checks, or explicit verification needs."
            )
        if "refactor" in role_set:
            role_policy_lines.append(
                "- Use `refactor` as a SECOND-PASS role for complex generation when first-pass code is likely suboptimal and needs cleanup/restructure/readability/performance improvement while preserving behavior."
            )
            role_policy_lines.append(
                "- Do NOT add `refactor` for simple one-pass tasks unless the query explicitly asks for optimization/refinement."
            )
        role_policy_lines.append(
            "- Prefer `single` with one role only for truly simple tasks; otherwise add roles only when justified by query complexity."
        )
        base_prompt = (
            "You are a meta-router for code development. Decide the best topology.\n"
            "Return JSON: {\"topology\": \"...\", \"roles\": [...], \"manager_role\": \"...\", "
            "\"entry_role\": \"...\", \"max_steps\": N}\n\n"
            f"Available roles: {roles_text}\n"
        )

    role_policy = "\n".join(role_policy_lines)

    if strategy == "single_preference":
        # Current behavior - strongly prefer single agent
        topology_guide = """
**Topology:**
1. "single" (DEFAULT): Simple funcs, basic algos, <15 lines
2. "centralized": Complex algos (DP/graphs), multi-phase, edge cases
3. "decentralized" (RARE): Parallel tasks, agents set next_role

**Roles:** single=["builder"], centralized=["planner","builder"](+checker,+refactor), decentralized=2-3 no mgr
**Steps:** single=1, centralized=2-3, decentralized=2-4
**Rules:** single: entry_role="builder" mgr=null | centralized: mgr+entry="planner" | decentralized: mgr=null entry=1st. Default single.
"""
    
    elif strategy == "balanced":
        # Neutral approach - decide based on actual complexity, aim for ~50-50 split
        topology_guide = """
**Topology:**
1. "single": Simple funcs, straightforward algos (<15 lines), no edge cases
2. "centralized": Algos with planning needs (DP/graphs/recursion), conditions, edge cases, multi-step
3. "decentralized": Parallel tasks, agents set next_role

**Roles:** single=["builder"], centralized=["planner","builder"](+checker,+refactor), decentralized=2-3 no mgr
**Steps:** single=1, centralized=2-4, decentralized=2-4
**Rules:** single: entry="builder" mgr=null | centralized: mgr+entry="planner" | decentralized: mgr=null entry=1st.
**Decision:** Choose based on complexity. Use single for straightforward implementations, centralized when planning/testing adds value.
"""
    
    elif strategy == "multi_preference":
        # Moderately prefer multi-agent, aim for ~70-80% centralized
        topology_guide = """
**Topology:**
1. "single": Only trivial funcs (<5 lines), no logic, hardcoded returns
2. "centralized" (RECOMMENDED): Standard for most algos, conditions, transforms, edge cases
3. "decentralized": Parallel components, agents set next_role

**Roles:** single=["builder"], centralized=["planner","builder"](+checker,+refactor), decentralized=2-3 no mgr
**Steps:** single=1, centralized=2-4, decentralized=2-4
**Rules:** single: entry="builder" mgr=null | centralized: mgr+entry="planner" | decentralized: mgr=null entry=1st.
**Decision:** Prefer centralized for better code quality and planning. Reserve single for truly trivial implementations.
"""
    
    elif strategy == "multi_aggressive":
        # Strongly prefer multi-agent for better quality, aim for >90% centralized
        topology_guide = """
**Topology:**
1. "single" (ALMOST NEVER): Only for hardcoded constants or trivial returns
2. "centralized" (MANDATORY): All algorithmic tasks, all logic, all transformations
3. "decentralized": Parallel independent components, agents set next_role

**Roles:** single=["builder"](avoid), centralized=ALWAYS["planner","builder","checker"](+refactor when needed), decentralized=3-4 parallel
**Steps:** single=1(rare), centralized=3-6, decentralized=3-6
**Rules:** single: entry="builder" mgr=null | centralized: mgr+entry="planner" | decentralized: mgr=null entry=1st.
**Decision:** Always use centralized for quality and robustness. Single only if implementation is completely trivial (e.g., return constant).
"""
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of: single_preference, balanced, multi_preference, multi_aggressive")
    
    return base_prompt + "\n" + role_policy + "\n\n" + topology_guide


def _normalize_role(value: Any) -> str:
    """Normalize role to lowercase string."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_topology_label(label: Dict[str, Any], available_roles: List[str]) -> Dict[str, Any]:
    if not isinstance(label, dict):
        label = {}

    topology = str(label.get("topology") or "").lower()
    if topology not in {"single", "centralized", "decentralized", "chain"}:
        topology = "single"

    roles = label.get("roles")
    manager_role = _normalize_role(label.get("manager_role"))
    entry_role = _normalize_role(label.get("entry_role"))
    max_steps = label.get("max_steps")

    # Backward-compat: {"roles": {"manager": "...", "workers": [...]}}
    if isinstance(roles, dict):
        mgr = _normalize_role(roles.get("manager"))
        workers = roles.get("workers") or []
        roles = []
        if mgr:
            roles.append(mgr)
        if isinstance(workers, list):
            roles.extend([_normalize_role(worker) for worker in workers])
        if manager_role is None:
            manager_role = mgr

    # Backward-compat: {"agent_id": "..."}
    if not roles:
        agent_id = _normalize_role(label.get("agent_id"))
        if agent_id:
            roles = [agent_id]

    if isinstance(roles, str):
        roles = [_normalize_role(roles)]
    if not isinstance(roles, list):
        roles = []

    roles = [_normalize_role(role) for role in roles]
    roles = [role for role in roles if role in available_roles]
    if not roles:
        roles = [available_roles[0]] if available_roles else ["builder"]

    if topology == "single":
        entry_role = entry_role or roles[0]
        if entry_role not in roles:
            entry_role = roles[0]
        roles = [entry_role]
        manager_role = None
        if max_steps is None:
            max_steps = 1
    elif topology == "decentralized":
        # Decentralized: no manager, agents work in parallel/sequentially with self-coordination
        manager_role = None
        entry_role = entry_role or roles[0]
        if entry_role not in roles:
            entry_role = roles[0]
        if max_steps is None:
            max_steps = min(6, max(2, len(roles)))
    elif topology == "centralized":
        # Centralized: manager coordinates workers
        if manager_role and manager_role not in roles:
            manager_role = None
        if not manager_role:
            manager_role = roles[0] if roles else None
        entry_role = entry_role or manager_role
        if max_steps is None:
            worker_count = max(1, len([r for r in roles if r != manager_role]))
            max_steps = min(3, worker_count + 1)
    else:
        # Other topologies (fallback)
        entry_role = entry_role or roles[0]
        if max_steps is None:
            max_steps = min(6, len(roles))

    return {
        "topology": topology,
        "roles": roles,
        "manager_role": manager_role,
        "entry_role": entry_role,
        "max_steps": int(max_steps) if max_steps else 1,
    }


def _is_complex_task(problem: Dict[str, Any]) -> bool:
    dataset = str(problem.get("dataset") or "").lower()
    if dataset == "gsm8k":
        prompt = str(problem.get("prompt") or "").lower()
        score = 0
        if len(prompt) >= 120:
            score += 1
        if len(re.findall(r"\d+(?:\.\d+)?", prompt)) >= 3:
            score += 1
        math_keywords = [
            "percent",
            "percentage",
            "ratio",
            "fraction",
            "average",
            "probability",
            "consecutive",
            "remaining",
            "left",
            "twice",
            "half",
            "triple",
            "difference",
            "total",
            "altogether",
            "per",
            "each",
            "combined",
        ]
        if any(keyword in prompt for keyword in math_keywords):
            score += 1
        return score >= 2
    if dataset == "math":
        prompt = str(problem.get("prompt") or "").lower()
        score = 0
        if len(prompt) >= 160:
            score += 1
        if len(re.findall(r"\\(frac|sum|prod|sqrt|mod|begin|end|cdot|equiv|pmod)", prompt)) >= 2:
            score += 1
        math_keywords = [
            "triangle",
            "circle",
            "polygon",
            "matrix",
            "vector",
            "congruent",
            "inverse",
            "mod",
            "probability",
            "perimeter",
            "area",
            "determinant",
            "least",
            "greatest",
            "prime",
            "divisor",
            "lcm",
            "gcd",
        ]
        if any(keyword in prompt for keyword in math_keywords):
            score += 1
        return score >= 2

    prompt = str(problem.get("prompt") or "").lower()
    tests_preview = str(problem.get("tests_preview") or "").lower()
    text = f"{prompt}\n{tests_preview}"

    score = 0
    if len(prompt) >= 110:
        score += 1
    if tests_preview.count("assert") >= 3:
        score += 1

    complexity_keywords = [
        "dynamic programming",
        "dp",
        "graph",
        "tree",
        "recursion",
        "binary search",
        "heap",
        "matrix",
        "subsequence",
        "subarray",
        "optimiz",
        "efficient",
        "minimum",
        "maximum",
        "shortest",
        "longest",
        "lcs",
        "inversion",
    ]
    if any(keyword in text for keyword in complexity_keywords):
        score += 1

    return score >= 2


def _maybe_add_refactor_role(
    label: Dict[str, Any], problem: Dict[str, Any], available_roles: List[str]
) -> Dict[str, Any]:
    if "refactor" not in available_roles:
        return label
    if not isinstance(label, dict):
        return label

    topology = str(label.get("topology") or "").lower()
    roles = label.get("roles")
    if not isinstance(roles, list):
        return label

    normalized_roles = [_normalize_role(r) for r in roles]
    if topology != "centralized":
        return label
    if "refactor" in normalized_roles:
        return label

    # Centralized + checker usually indicates a second-pass polish opportunity.
    # For other centralized cases, only add refactor when the task looks complex.
    should_add = ("checker" in normalized_roles) or _is_complex_task(problem)
    if not should_add:
        return label

    normalized_roles.append("refactor")
    label["roles"] = normalized_roles
    if label.get("max_steps") is None or int(label.get("max_steps") or 0) < 3:
        label["max_steps"] = 3
    return label


def _parse_and_normalize_label(
    response_text: str,
    available_roles: List[str],
    problem: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
    label_json = json.loads(cleaned_response)
    label_json = _normalize_topology_label(label_json, available_roles)
    label_json = _maybe_add_refactor_role(label_json, problem, available_roles)
    return label_json


def load_problems(file_paths: List[str], dataset: str = "humaneval") -> List[Dict[str, Any]]:
    problems = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id", "") or data.get("name", "") or data.get("id", "")
                    prompt_text = ""
                    entry_point = ""
                    tests_preview = ""

                    if dataset == "mbpp":
                        prompt_text = str(data.get("text") or data.get("prompt") or "")
                        code_text = str(data.get("code") or "")
                        test_list = data.get("test_list") or []
                        entry_point = (
                            str(data.get("entry_point") or "")
                            or _extract_entry_point_from_tests(test_list if isinstance(test_list, list) else [])
                            or _extract_entry_point_from_code(code_text)
                        )
                        if isinstance(test_list, list) and test_list:
                            tests_preview = "\n".join([str(x) for x in test_list[:3]])
                    elif dataset in {"gsm8k", "math"}:
                        prompt_text = str(data.get("question") or data.get("prompt") or data.get("task") or "")
                    else:
                        # humaneval
                        prompt_text = str(data.get("prompt") or "")
                        entry_point = str(data.get("entry_point") or "")

                    if prompt_text:
                        problems.append({
                            "task_id": task_id,
                            "prompt": prompt_text,
                            "entry_point": entry_point,
                            "tests_preview": tests_preview,
                            "dataset": dataset,
                            "original_data": data
                        })
                except json.JSONDecodeError:
                    continue
    return problems

def generate_labels(
    problems: List[Dict[str, Any]],
    output_file: str,
    model: str,
    available_roles: List[str],
    strategy: str = "single_preference",
    dataset: str = "humaneval",
):
    # Initialize Client (Start your stronger model here, e.g. GPT-4)
    # Ensure LLM_API_KEY and LLM_BASE_URL are set in environment
    client = LLMClient(model=model)
    
    # Get system prompt based on strategy
    system_prompt = get_system_prompt(strategy, available_roles, dataset=dataset)
    
    print(f"DEBUG: Using LLM API Base URL: {client.base_url}")
    print(f"Starting label generation for {len(problems)} problems using {model}...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, prob in enumerate(problems):
            if dataset in {"gsm8k", "math"}:
                task_text = (
                    "Task: Solve the following math word problem and provide final answer as #### <number>.\n\n"
                    f"Problem: {prob['prompt']}"
                )
            else:
                task_text = f"Task: {prob['prompt']}"
                if prob['entry_point']:
                    task_text += f"\nEntry point: {prob['entry_point']}"
                if prob.get("tests_preview"):
                    task_text += f"\nTests (preview):\n{prob['tests_preview']}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_text}
            ]

            try:
                print(f"[{i+1}/{len(problems)}] Processing {prob['task_id']}...", end="", flush=True)
                
                # Call Teacher Model
                # We use JSON mode if supported by the provider, otherwise normal
                response_text = client.chat(
                    messages=messages, 
                    temperature=0.1,
                    # response_format={"type": "json_object"} # Uncomment if using OpenAI GPT-4-1106+
                )
                label_json = _parse_and_normalize_label(response_text, available_roles, prob)

                # Construct Training Sample
                training_sample = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task_text},
                        {"role": "assistant", "content": json.dumps(label_json, ensure_ascii=False)}
                    ],
                    "sample_type": f"{dataset}_teacher_distilled",
                    "origin_task_id": prob['task_id']
                }

                f_out.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
                f_out.flush()
                print(" Done.")

            except Exception as e:
                if _is_content_filter_error(e):
                    try:
                        safe_task_text = _build_sanitized_task_text(prob, dataset)
                        safe_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": safe_task_text},
                        ]
                        response_text = client.chat(
                            messages=safe_messages,
                            temperature=0.1,
                        )
                        label_json = _parse_and_normalize_label(response_text, available_roles, prob)
                        training_sample = {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": safe_task_text},
                                {"role": "assistant", "content": json.dumps(label_json, ensure_ascii=False)}
                            ],
                            "sample_type": f"{dataset}_teacher_distilled",
                            "origin_task_id": prob['task_id'],
                            "label_mode": "sanitized_fallback",
                        }
                        f_out.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
                        f_out.flush()
                        print(" Done (sanitized fallback).")
                    except Exception as e2:
                        print(f" Failed: {e2}")
                else:
                    print(f" Failed: {e}")

    print(f"Label generation complete. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Router Labels using a Teacher LLM")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path(s) to dataset JSONL, comma-separated if multiple",
    )
    parser.add_argument("--output", type=str, default="humaneval_router_labels.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Teacher model name")
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp", "gsm8k", "math"],
        help="Input dataset format.",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="builder,planner,checker,refactor",
        help="comma-separated available roles to constrain the label output",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="single_preference",
        choices=["single_preference", "balanced", "multi_preference", "multi_aggressive"],
        help="Agent topology preference strategy: "
             "single_preference (default, strongly prefer single agent), "
             "balanced (neutral, choose based on complexity), "
             "multi_preference (moderately prefer multi-agent), "
             "multi_aggressive (strongly prefer multi-agent collaboration)",
    )
    
    args = parser.parse_args()
    
    data_paths = _parse_list(args.data) or [args.data]
    probs = load_problems(data_paths, dataset=args.dataset)
    roles = [role.lower() for role in _parse_list(args.roles)]
    generate_labels(probs, args.output, args.model, roles, args.strategy, dataset=args.dataset)
