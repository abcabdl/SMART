import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str
    entry_point: str
    test: str

def load_humaneval_data(file_paths: List[str]) -> List[HumanEvalProblem]:
    """Load HumanEval problems from jsonl files."""
    problems = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Support both standard HumanEval and the format provided in the prompt
                    task_id = data.get("task_id") or data.get("name")
                    prompt = data.get("prompt")
                    entry_point = data.get("entry_point")
                    test_code = data.get("test")
                    
                    if task_id and prompt:
                        problems.append(HumanEvalProblem(
                            task_id=task_id,
                            prompt=prompt,
                            entry_point=entry_point,
                            test=test_code
                        ))
                except json.JSONDecodeError:
                    continue
    return problems

def generate_router_training_data(
    problems: List[HumanEvalProblem], 
    output_path: str,
    topology_strategy: str = "dynamic" 
):
    """
    Convert HumanEval problems into Router SFT training data.
    
    Args:
        problems: List of HumanEval problems
        output_path: Where to save the jsonl file
        topology_strategy: 'dynamic' (mix of simple/complex) or 'fixed'
    """
    
    sft_data = []
    
    system_prompt = (
        "You are an expert software architect and team lead. "
        "Analyze the coding task and determine the best team structure (topology) "
        "and assign specific roles to specialized agents to solve it efficiently."
    )

    for prob in problems:
        # Heuristic to determine complexity
        # Loosely based on prompt length and keywords
        is_complex = len(prob.prompt) > 200 or "class" in prob.prompt or "complex" in prob.prompt
        
        user_content = f"Task: Implement the following Python function:\n\n{prob.prompt}"
        
        if is_complex and topology_strategy == "dynamic":
            # For complex tasks -> Centralized Team
            # This teaches the model to use a team for harder problems
            topology = "centralized"
            roles = ["Planner", "PythonDeveloper", "TestEngineer"]
            reasoning = (
                "The task involves complex logic or multiple requirements. "
                "A Planner is needed to break down edge cases. "
                "A Developer to implement the logic. "
                "A Tester to verify against the provided examples."
            )
            structure = {
                "reasoning": reasoning,
                "topology": "centralized",
                "roles": {
                    "manager": "Planner",
                    "workers": ["PythonDeveloper", "TestEngineer"]
                }
            }
        else:
            # For simple tasks -> Single Agent
            # Teaches the model to be efficient
            topology = "single"
            roles = ["PythonDeveloper"]
            reasoning = "The task is a straightforward function implementation. A single developer is sufficient."
            structure = {
                "reasoning": reasoning,
                "topology": "single",
                "agent_id": "PythonDeveloper"
            }

        # Construct the conversation turn
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(structure, ensure_ascii=False)}
            ],
            "sample_type": "humaneval_routing"
        }
        sft_data.append(conversation)

    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(sft_data)} training samples at {output_path}")

if __name__ == "__main__":
    # Example usage (you would run this to prepare data before training)
    # import sys
    # if len(sys.argv) > 1:
    #     data = load_humaneval_data([sys.argv[1]])
    #     generate_router_training_data(data, "router_sft_humaneval.jsonl")
    pass
