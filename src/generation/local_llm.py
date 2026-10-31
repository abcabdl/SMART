from __future__ import annotations

import copy
import gc
import json
import os
from typing import Any, Dict, Iterable, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _require(pkg: str, install_hint: Optional[str] = None) -> None:
    try:
        __import__(pkg)
    except ImportError as exc:  # pragma: no cover
        hint = install_hint or f"pip install {pkg}"
        raise SystemExit(
            f"Missing dependency '{pkg}'. Install with: {hint}"
        ) from exc


_require("transformers", "pip install transformers")

import torch  # noqa: E402
try:  # noqa: E402
    from peft import PeftModel  # type: ignore
except ImportError:  # pragma: no cover
    PeftModel = None  # type: ignore[assignment]
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _resolve_dtype(value: str) -> Optional[torch.dtype]:
    value = (value or "").lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    return None


def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _safe_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _safe_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _maybe_disable_transformers_cuda_warmup() -> None:
    # transformers may allocate a large temporary block in caching_allocator_warmup.
    # On 8GB-class GPUs this can trigger OOM even when final placement would fit.
    if not _safe_bool_env("LOCAL_LLM_DISABLE_CUDA_WARMUP", True):
        return
    try:
        import transformers.modeling_utils as modeling_utils  # type: ignore

        if getattr(modeling_utils, "_local_llm_warmup_patched", False):
            return

        def _no_warmup(*_args: Any, **_kwargs: Any) -> None:
            return

        setattr(modeling_utils, "caching_allocator_warmup", _no_warmup)
        setattr(modeling_utils, "_local_llm_warmup_patched", True)
    except Exception:
        # If patching fails, continue with normal flow.
        return


def _build_cuda_max_memory(
    reserve_mib: int = 256,
    min_assign_mib: int = 256,
    utilization_ratio: float = 0.90,
) -> Dict[Any, str]:
    if not torch.cuda.is_available():
        return {}
    max_memory: Dict[Any, str] = {}
    reserve_bytes = max(0, int(reserve_mib)) * 1024 * 1024
    min_assign_bytes = max(0, int(min_assign_mib)) * 1024 * 1024
    for gpu_idx in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(gpu_idx):
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
        except Exception:
            continue
        budget_bytes = int(float(free_bytes) * float(utilization_ratio)) - reserve_bytes
        if budget_bytes <= min_assign_bytes:
            continue
        max_memory[gpu_idx] = f"{max(1, budget_bytes // (1024 * 1024))}MiB"
    return max_memory


def _scale_gpu_max_memory(max_memory: Dict[Any, Any], scale: float, floor_mib: int = 1024) -> Dict[Any, Any]:
    scaled: Dict[Any, Any] = {}
    factor = max(0.10, min(1.0, float(scale)))
    floor = max(128, int(floor_mib))
    for key, value in (max_memory or {}).items():
        if key == "cpu":
            scaled[key] = value
            continue
        mib: Optional[int] = None
        if isinstance(value, str):
            text = value.strip().lower()
            if text.endswith("mib"):
                try:
                    mib = int(text[:-3].strip())
                except Exception:
                    mib = None
        elif isinstance(value, (int, float)):
            mib = int(float(value) / (1024.0 * 1024.0))

        if mib is None:
            scaled[key] = value
            continue
        scaled_mib = max(floor, int(mib * factor))
        scaled[key] = f"{scaled_mib}MiB"
    return scaled


def _parse_retry_plan(plan: str) -> list[tuple[str, float]]:
    # Format: "auto:0.82,balanced:0.72,cpu:1.0"
    items: list[tuple[str, float]] = []
    raw = str(plan or "").strip()
    if not raw:
        return items
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        if ":" in part:
            name, scale = part.split(":", 1)
            try:
                items.append((name.strip().lower(), float(scale.strip())))
            except Exception:
                items.append((name.strip().lower(), 0.8))
        else:
            items.append((part.lower(), 0.8))
    return items


def _resolve_retry_device_map(name: str) -> Any:
    key = str(name or "").strip().lower()
    if key in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return key
    if key in {"cpu", "offload_cpu"}:
        return {"": "cpu"}
    return "auto"


def _normalize_quantized_device_map(
    requested_device: Optional[str],
    gpu_count: int,
) -> Any:
    env_device_map = os.getenv("LOCAL_LLM_DEVICE_MAP")
    if env_device_map:
        text = env_device_map.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except Exception:
                return text
        return text

    normalized = str(requested_device or "").strip().lower()
    if normalized.startswith("cuda:"):
        try:
            gpu_idx = int(normalized.split(":", 1)[1])
            return {"": gpu_idx}
        except (TypeError, ValueError):
            pass
    if normalized in {"cpu", "mps"}:
        return {"": normalized}
    if gpu_count > 1:
        # Prefer true multi-GPU balancing by default.
        # "balanced_low_0" intentionally biases away from cuda:0 and can look like single-GPU usage.
        return os.getenv("LOCAL_LLM_MULTI_GPU_STRATEGY", "balanced").strip() or "balanced"
    return "auto"


def _resolve_generation_input_device(model: Any, fallback_device: str) -> str:
    # For sharded models, route inputs to the first CUDA shard to avoid accidental single-device pinning.
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, int):
                return f"cuda:{value}"
            if isinstance(value, str):
                lowered = value.lower()
                if lowered.startswith("cuda"):
                    return value
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return str(model_device)
    return fallback_device


def _debug_load_kwargs(enabled: bool, kwargs: Dict[str, Any]) -> None:
    if not enabled:
        return
    payload = {
        "device_map": kwargs.get("device_map"),
        "max_memory": kwargs.get("max_memory"),
        "offload_folder": kwargs.get("offload_folder"),
        "low_cpu_mem_usage": kwargs.get("low_cpu_mem_usage"),
        "offload_buffers": kwargs.get("offload_buffers"),
    }
    print(f"[LocalLLMClient] load_kwargs={payload}", flush=True)


class LocalLLMClient:
    """Local transformers + LoRA inference client with OpenAI-like chat() API."""

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_4bit: bool = False,
        use_8bit: bool = False,
    ) -> None:
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = _resolve_dtype(dtype)
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        debug_load = bool(str(os.getenv("LOCAL_LLM_DEBUG_LOAD", "")).strip().lower() in {"1", "true", "yes"})
        if _safe_bool_env("LOCAL_LLM_DISABLE_PARALLEL_LOADING", True):
            # Reduce transient VRAM spikes from concurrent tensor materialization on small GPUs.
            os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
            os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
        if use_4bit or use_8bit:
            _require("bitsandbytes", "pip install bitsandbytes")
            _maybe_disable_transformers_cuda_warmup()
            reserve_mib = _safe_int_env("LOCAL_LLM_GPU_RESERVE_MIB", 256)
            min_assign_mib = _safe_int_env("LOCAL_LLM_GPU_MIN_ASSIGN_MIB", 256)
            utilization_ratio = _safe_float_env("LOCAL_LLM_GPU_UTILIZATION_RATIO", 0.90)
            cpu_max_memory = os.getenv("LOCAL_LLM_CPU_MAX_MEMORY", "48GiB")
            cuda_max_memory = _build_cuda_max_memory(
                reserve_mib=reserve_mib,
                min_assign_mib=min_assign_mib,
                utilization_ratio=utilization_ratio,
            )
            quant_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                load_in_8bit=use_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype is torch.bfloat16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = _normalize_quantized_device_map(self.device, torch.cuda.device_count())
            model_kwargs["low_cpu_mem_usage"] = True
            model_kwargs["offload_state_dict"] = True
            model_kwargs["offload_buffers"] = True
            model_kwargs["offload_folder"] = os.getenv("LOCAL_LLM_OFFLOAD_DIR", "offload")
            if cuda_max_memory:
                # Keep GPU keys as integers for broad accelerate compatibility.
                cuda_max_memory["cpu"] = cpu_max_memory
                model_kwargs["max_memory"] = cuda_max_memory
        elif torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        _debug_load_kwargs(debug_load, model_kwargs)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except torch.OutOfMemoryError as first_oom:
            # Progressive fallback for tight multi-GPU VRAM budgets.
            if use_4bit or use_8bit:
                retry_plan = _parse_retry_plan(
                    os.getenv(
                        "LOCAL_LLM_OOM_RETRY_PLAN",
                        "auto:0.72,balanced:0.64,sequential:0.56,cpu:1.0",
                    )
                )
                retry_floor_mib = _safe_int_env("LOCAL_LLM_OOM_RETRY_MIN_MIB", 256)
                last_oom: BaseException = first_oom
                model = None  # type: ignore[assignment]
                base_kwargs = copy.deepcopy(model_kwargs)
                for retry_name, retry_scale in retry_plan:
                    fallback_kwargs = copy.deepcopy(base_kwargs)
                    fallback_kwargs["device_map"] = _resolve_retry_device_map(retry_name)
                    if "max_memory" in fallback_kwargs and isinstance(fallback_kwargs["max_memory"], dict):
                        fallback_kwargs["max_memory"] = _scale_gpu_max_memory(
                            fallback_kwargs["max_memory"],
                            scale=retry_scale,
                            floor_mib=retry_floor_mib,
                        )
                        if fallback_kwargs["device_map"] == {"": "cpu"}:
                            cpu_only = {"cpu": fallback_kwargs["max_memory"].get("cpu", "64GiB")}
                            fallback_kwargs["max_memory"] = cpu_only
                    _debug_load_kwargs(debug_load, fallback_kwargs)
                    try:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                        model = AutoModelForCausalLM.from_pretrained(model_path, **fallback_kwargs)
                        break
                    except torch.OutOfMemoryError as retry_oom:
                        last_oom = retry_oom
                        continue
                if model is None:
                    raise last_oom
            else:
                raise
        if lora_path:
            if PeftModel is None:
                raise SystemExit("Missing dependency 'peft'. Install with: pip install peft")
            model = PeftModel.from_pretrained(model, lora_path)

        if not (use_4bit or use_8bit):
            model = model.to(self.device)

        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def _build_prompt(self, messages: Iterable[Dict[str, str]]) -> str:
        messages_list = list(messages)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts = []
        for msg in messages_list:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if self.use_4bit or self.use_8bit:
            input_device = _resolve_generation_input_device(self.model, self.device)
            input_ids = input_ids.to(input_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_device)
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        gen_temperature = temperature if temperature is not None else self.temperature
        max_new_tokens = max_tokens or self.max_new_tokens
        do_sample = gen_temperature > 0

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=gen_temperature if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
