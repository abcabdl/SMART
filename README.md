# SMART: Scalable Multi-Agent Role-conditioned Teaming via LLM-free Tree Search

Official code for **SMART** (NeurIPS 2026, oral).

SMART assembles a team of agents for each query from a large pool of specialized agents. It works in three stages:

1. **Task-conditioned topology planning.** An LLM planner chooses a collaboration topology and the set of roles (`planner`, `builder`, `checker`, `refactor`, ...) for the query.
2. **Role-conditioned candidate construction.** For each role, an artifact-aware query is built and used to retrieve agents from a FAISS index. Constraint filtering and MMR then reduce them to a small candidate set of `K_f` agents.
3. **LLM-free MCTS role filling.** A Monte Carlo Tree Search assigns agents to roles online. The search scores teams with a metadata-only reward (unary quality, role transition, I/O-chain compatibility and artifact-gap reduction). Neither the search nor its rollouts call an LLM or execute tools.

The selected agents then run through the registered tool library. After each step the artifact bag and the gap vector are updated. The episode stops as soon as the task's artifacts are complete.

## Repository layout

```
configs/smart_pdf_benchmark.json   # paper defaults (roles, retrieval, MCTS, reward, runtime, asset paths)
scripts/
  run_paper_{humaneval,mbpp,gsm8k,math}.py   # main benchmark entry points (paper preset applied)
  build_assets/                               # regenerate the agent/tool ecosystem and indexes
src/
  core/         # agent/tool cards, SQLite registry, constraints, role-query builder, topology types
  retrieval/    # embedders, FAISS HNSW index, role-conditioned search service, MMR
  routing/      # smart_reward.py (Eqs. 2-6), smart_mcts.py (LLM-free MCTS), bandit store
  execution/    # orchestrator (topology planning, role execution, artifact/gap tracking), sandboxed tool executor
  generation/   # LLM client, local HF model wrapper, programmatic agent/tool catalogs
  evaluation/   # HumanEval / MBPP / GSM8K / MATH / HotpotQA evaluators and the paper preset
  analysis/     # analysis experiments (Pareto, pool-size scalability, teaming statistics, proxy-reward correlation)
tests/          # unit and contract tests
data/datas/     # benchmark splits ({humaneval,mbpp,gsm8k,math,hotpotqa}_{test,validate}.jsonl)
```

Prebuilt assets at the repository root:

| Asset | Contents | Index |
|---|---|---|
| `paper_registry.sqlite` | **The 300-agent / 1,500-tool ecosystem from the paper** (75 agents × 4 benchmark families, 5 tools per agent) | `index_paper/` (dummy embedder, dim 64) |
| `real_dataset_registry.sqlite` | The same ecosystem plus 75 HotpotQA agents (375 agents / 1,875 tools); default for the run scripts | `index_real_dataset/` (all-MiniLM-L6-v2, dim 384), `index_real_dataset_dummy/` (dim 64) |
| `hotpotqa_appendix_registry.sqlite` | 75 multi-hop QA agents used for HotpotQA | `index_hotpotqa_appendix/` (dim 64) |
| `postprocess_registry.sqlite` | 9 deterministic code post-processing tools used by the HumanEval/MBPP evaluators (readable copies in `generated_tools/`) | – |

Tool source code is stored inside each registry (`tool_code` table). The `generated_tools_*` directories hold readable copies of the same code. `*_agent_manifest.json` describes each ecosystem.

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/abcabdl/SMART.git
cd SMART
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`sentence-transformers` (and therefore PyTorch) is needed for the semantic index `index_real_dataset/`. Without it, the run scripts fall back to the dummy embedder and `index_real_dataset_dummy/`.

## LLM backend

Execution-time roles call an OpenAI-compatible chat API. Configure it through environment variables:

```bash
export LLM_API_KEY=sk-...                         # or OPENAI_API_KEY
export LLM_API_BASE=https://api.openai.com/v1     # or OPENAI_BASE_URL; any OpenAI-compatible endpoint
export LLM_MODEL=gpt-4o-mini                      # paper backbone
```

The evaluators also accept `--code_model / --code_base_url / --code_api_key`. To use a local Hugging Face model instead, pass `--local_model_dir <path>`. Raw LLM calls are logged to `runs/llm_raw.jsonl`. You can change that path with `LLM_RAW_LOG_PATH`.

The MCTS search makes no LLM calls (Stage 3). LLMs are used only for topology planning and for role execution.

## Quick start

A single query with a mock LLM needs no API key. It exercises planning, retrieval, MCTS selection and tool execution end to end:

```bash
python -m src.execution.run_query --paper_mode --use_mock_llm \
  --db real_dataset_registry.sqlite --index_dir index_real_dataset_dummy \
  --embedder dummy --dim 64 --roles planner,builder,checker,refactor \
  --dynamic_topology --mcts_dynamic_optimization \
  --query "Implement a function that returns the maximum subarray sum."
```

Drop `--use_mock_llm` and set the variables above to use a real model. With `sentence-transformers` installed, use `--index_dir index_real_dataset --embedder sentence-transformer --dim 384` instead.

## Reproducing the main results

The four main benchmarks use the paper preset in `configs/smart_pdf_benchmark.json`. Each script adds the preset arguments, and any extra flags are forwarded to the evaluator:

```bash
python scripts/run_paper_humaneval.py --tasks data/datas/humaneval_test.jsonl --out runs/humaneval_report.json
python scripts/run_paper_mbpp.py      --mbpp_tasks data/datas/mbpp_test.jsonl    --out runs/mbpp_report.json
python scripts/run_paper_gsm8k.py     --tasks data/datas/gsm8k_test.jsonl     --out runs/gsm8k_report.json
python scripts/run_paper_math.py      --tasks data/datas/math_test.jsonl      --out runs/math_report.json
```

To run against the paper's 300-agent ecosystem without the HotpotQA agents, add `--db paper_registry.sqlite --index_dir index_paper --embedder dummy --dim 64`.

HotpotQA uses `python -m src.evaluation.eval_hotpotqa` (see `--help`; it defaults to `data/datas/hotpotqa_validate.jsonl`).

### Paper hyperparameters

| Component | Setting (`configs/smart_pdf_benchmark.json`) |
|---|---|
| Roles | `planner, builder, checker, refactor` (MBPP starts without `refactor` and uses it only for repair) |
| Retrieval | `top_n=30`, `top_k=15`, MMR `λ=0.5`, artifact-aware role queries |
| MCTS | `K_f=5`, budget `B=64`, depth `H=4`, `C=1.414`, `γ=0.95` |
| Reward | `mean_active` over unary, transition, I/O-chain and gap-reduction terms |
| Runtime | seed 7; cross-instance cache, cross-instance posterior updates and parallel frontier all **off** |

Ablation switches exposed by the evaluators include `--no_transition_term`, `--no_iochain`, `--no_gap_term`, `--no_mmr`, `--reward_variant`, `--mcts_iterations`, `--mcts_rollout_depth`, `--mcts_max_candidates` and `--use_parallel_frontier`.

### Where the method lives in the code

| Paper | Code |
|---|---|
| Stage 1: topology planning | `src/execution/orchestrator.py` (`_plan_topology`, `_heuristic_topology`), `src/core/topology.py` |
| Stage 2: role-conditioned retrieval + MMR | `src/core/query_builder.py` (`build_role_query`), `src/retrieval/search_service.py` (`get_candidates`), `src/retrieval/mmr.py` |
| Reward, Eqs. (2)–(6) | `src/routing/smart_reward.py` |
| Stage 3: LLM-free MCTS | `src/routing/smart_mcts.py` (`select_with_smart_mcts`) |
| Execution and online update | `src/execution/orchestrator.py`, `src/execution/tool_executor.py`, `src/execution/sandbox.py`, `src/routing/bandit_store.py` |
| Agent metadata schema | `src/core/cards.py` |

## Agent ecosystem

The ecosystem is synthesized programmatically. Agent cards and tool specs come from `src/generation/real_dataset_bulk_catalog.py` and are registered with schema validation. To regenerate:

```bash
python scripts/build_assets/generate_real_dataset_assets.py      # real_dataset_registry.sqlite + index + manifest
python scripts/build_assets/generate_paper_assets.py             # paper_registry.sqlite (300 agents) + index_paper
python scripts/build_assets/generate_hotpotqa_appendix_assets.py # HotpotQA registry + index
python scripts/build_assets/generate_paper_alignment_report.py   # reports/paper_alignment_report.json
```

To rebuild an index only:

```bash
python -m src.retrieval.build_index --db real_dataset_registry.sqlite --kind agent \
  --out index_real_dataset --embedder sentence-transformer --dim 384
```

Tools are invoked as `run(inputs: dict) -> dict` inside a sandbox with a timeout and module allow-list (`src/execution/sandbox.py`).

## Tests

```bash
pytest tests
```

Two legacy tests (`test_index.py::test_build_index_and_search`, `test_mbpp_eval_repairs.py::test_mbpp_failure_context_includes_return_risk_and_rewrite_mode`) are marked `xfail`. Their expected strings predate later changes to the dummy embedder and the MBPP repair prompt.

## Citation

```bibtex
@inproceedings{zhang2026smart,
  title     = {{SMART}: Scalable Multi-Agent Role-conditioned Teaming via {LLM}-free Tree Search},
  author    = {Zhang, Runzhe and Li, Ning and Zeng, Xingshan and Chen, Letian and Wang, Yasheng and Zhang, Weinan and Yu, Yong and Liu, Weiwen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE). Benchmark files under `data/datas/` keep the licenses of the original datasets (HumanEval, MBPP, GSM8K, MATH, HotpotQA).
