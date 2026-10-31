from __future__ import annotations

import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"Missing dependency '{pkg}'. Install training deps first: "
            "pip install transformers datasets peft accelerate bitsandbytes"
        ) from exc


_require("transformers")
_require("datasets")
_require("peft")

import torch  # noqa: E402
from datasets import concatenate_datasets, load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _extract_entry_point_from_row(example: Dict[str, Any]) -> str:
    entry = str(example.get("entry_point") or "")
    if entry:
        return entry
    code = str(example.get("code") or "")
    fn = _extract_first_function_from_code(code)
    if fn is not None:
        return fn.name
    m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    return m.group(1) if m else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune router model")
    parser.add_argument("--data", type=str, required=True, help="router_sft.jsonl (comma-separated for multiple)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument(
        "--download_model",
        action="store_true",
        help="download HF model weights to local dir before fine-tuning",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/hf_cache",
        help="local directory to store downloaded HF weights",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (or set HUGGINGFACE_HUB_TOKEN)",
    )
    parser.add_argument("--output_dir", type=str, default="models/router_lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_types", type=str, default="", help="comma-separated sample_type filter")
    parser.add_argument("--train_on_inputs", action="store_true", help="include prompt tokens in loss")
    parser.add_argument(
        "--drop_no_output",
        action="store_true",
        help="drop examples without target output (recommended for HumanEval-style data)",
    )
    parser.add_argument(
        "--convert_humaneval",
        action="store_true",
        help="automatically convert raw HumanEval rows (name, prompt, test) into router training samples with heuristic labels",
    )
    parser.add_argument(
        "--convert_dataset",
        type=str,
        default="none",
        choices=["none", "humaneval", "mbpp"],
        help="convert raw dataset rows to router chat samples before training",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_files = _parse_list(args.data)
    if not data_files:
        raise SystemExit("No training data provided.")
    datasets = [load_dataset("json", data_files=path, split="train") for path in data_files]
    dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    sample_types = set(_parse_list(args.sample_types))
    if sample_types:
        dataset = dataset.filter(lambda row: row.get("sample_type") in sample_types)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    convert_dataset = args.convert_dataset
    if args.convert_humaneval and convert_dataset == "none":
        convert_dataset = "humaneval"

    if convert_dataset != "none":
        def _heuristic_label_generator(prompt_text: str) -> Dict[str, Any]:
            # Simple heuristic to generate training targets
            # Ideally this should come from a teacher model (e.g. GPT-4)
            needs_test = "test" in prompt_text.lower() or "assert" in prompt_text.lower()
            needs_math = "math" in prompt_text.lower() or "calculate" in prompt_text.lower()
            
            roles = ["PythonDeveloper"]
            if needs_test:
                roles.append("TestEngineer")
            if needs_math:
                roles.append("MathExpert")
                
            topology = "centralized" if len(roles) > 1 else "single"
            
            return {
                "reasoning": "Based on task requirements (coding + potential testing/math).",
                "topology": topology,
                "roles": {
                    "manager": "Planner",
                    "workers": roles
                } if topology == "centralized" else {},
                "agent_id": roles[0] if topology == "single" else None
            }

        def _convert_dataset_row(example: Dict[str, Any]) -> Dict[str, Any]:
            # If already has messages, skip
            if example.get("messages"):
                return example

            if convert_dataset == "mbpp":
                task_prompt = str(example.get("text") or example.get("prompt") or "")
                entry_point = _extract_entry_point_from_row(example)
                tests = example.get("test_list") or []
                tests_preview = ""
                if isinstance(tests, list) and tests:
                    tests_preview = "\n".join([str(x) for x in tests[:3]])
            else:
                # humaneval
                task_prompt = str(example.get("prompt") or example.get("text") or "")
                entry_point = str(example.get("entry_point") or "")
                tests_preview = str(example.get("test") or "")

            user_content = f"Task: Implement the following Python function:\n\n{task_prompt}"
            if entry_point:
                user_content += f"\n\nEntry point: {entry_point}"
            if tests_preview:
                user_content += f"\n\nTests (preview):\n{tests_preview}"

            structure = _heuristic_label_generator(user_content)
            
            system_prompt = (
                "You are an expert software architect. "
                "Analyze the coding task and determine the best team structure (topology) "
                "and assign specific roles to specialized agents to solve it efficiently."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(structure, ensure_ascii=False)}
            ]
            return {
                "messages": messages,
                "converted": True,
                "sample_type": f"{convert_dataset}_heuristic_distilled",
            }

        dataset = dataset.map(_convert_dataset_row)

    def _extract_prompt_response(example: Dict[str, Any]) -> Dict[str, str]:
        prompt = str(example.get("input") or "")
        response = str(example.get("output") or "")
        if example.get("prompt") is not None:
            prompt = str(example.get("prompt") or "")
            response = (
                example.get("completion")
                or example.get("solution")
                or example.get("code")
                or example.get("output")
                or ""
            )
            response = str(response)
        return {"prompt": prompt, "response": response}

    if args.drop_no_output:
        def _has_output(example: Dict[str, Any]) -> bool:
            messages = example.get("messages")
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except json.JSONDecodeError:
                    messages = None
            if isinstance(messages, list) and messages:
                return True
            payload = _extract_prompt_response(example)
            return bool(payload["response"].strip())

        dataset = dataset.filter(_has_output)

    model_path = args.model
    if args.download_model and not os.path.isdir(model_path):
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency 'huggingface_hub'. Install it with: pip install huggingface_hub"
            ) from exc
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = snapshot_download(
            repo_id=args.model,
            local_dir=args.model_dir,
            local_dir_use_symlinks=False,
            token=args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize_prompt_response(prompt: str, response: str) -> Dict[str, Any]:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = tokenizer(response, add_special_tokens=False).input_ids
        input_ids = prompt_ids + response_ids
        if args.train_on_inputs:
            labels = list(input_ids)
        else:
            labels = [-100] * len(prompt_ids) + response_ids
        return {"input_ids": input_ids, "labels": labels}

    def _tokenize_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not hasattr(tokenizer, "apply_chat_template"):
            joined = "\n".join(f"{msg.get('role')}: {msg.get('content','')}" for msg in messages)
            return _tokenize_prompt_response("", joined)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if args.train_on_inputs:
            labels = list(full_ids)
        else:
            labels = [-100] * len(full_ids)
            prev_len = 0
            for idx in range(len(messages)):
                partial_text = tokenizer.apply_chat_template(
                    messages[: idx + 1],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                partial_ids = tokenizer(partial_text, add_special_tokens=False).input_ids
                added_len = len(partial_ids) - prev_len
                if messages[idx].get("role") == "assistant":
                    labels[prev_len : prev_len + added_len] = partial_ids[prev_len : prev_len + added_len]
                prev_len = len(partial_ids)
        return {"input_ids": full_ids, "labels": labels}

    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example.get("messages")
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                messages = None
        if isinstance(messages, list) and messages:
            payload = _tokenize_messages(messages)
        else:
            pair = _extract_prompt_response(example)
            payload = _tokenize_prompt_response(pair["prompt"], pair["response"])

        input_ids = payload["input_ids"][: args.max_length]
        labels = payload["labels"][: args.max_length]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    quant_config = None
    model_kwargs = {"trust_remote_code": True}
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_distributed = local_rank >= 0
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quant_config
        # For DDP + 4-bit, each rank must load the model on its own device.
        if is_distributed:
            model_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if args.use_4bit:
        # prepare_model_for_kbit_training may enable checkpointing internally.
        # Control it explicitly to avoid double-enabling in DDP.
        if args.gradient_checkpointing:
            try:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except TypeError:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=True,
                )
        else:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=False,
            )
    elif args.gradient_checkpointing:
        # DDP + LoRA often needs non-reentrant checkpointing to avoid
        # "Expected to mark a variable ready only once" runtime errors.
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    target_modules = _parse_list(args.target_modules)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    pad_id = tokenizer.pad_token_id

    def collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            ids = list(item["input_ids"])
            mask = list(item["attention_mask"])
            lab = list(item["labels"])
            pad_len = max_len - len(ids)
            if pad_len:
                ids.extend([pad_id] * pad_len)
                mask.extend([0] * pad_len)
                lab.extend([-100] * pad_len)
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lab)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=[],
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        # Checkpointing has been enabled on the model explicitly above.
        gradient_checkpointing=False,
        # Avoid extra autograd traversals and DDP hook conflicts with LoRA+checkpointing.
        ddp_find_unused_parameters=False if is_distributed else None,
        gradient_checkpointing_kwargs=None,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta_path = os.path.join(args.output_dir, "router_lora_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"base_model": args.model, "resolved_model_path": model_path}, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
