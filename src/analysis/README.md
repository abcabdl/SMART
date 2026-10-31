# MBPP Analysis Scripts

All analysis code is isolated under `src/analysis` and does not change default evaluation behavior.

## Scripts

- `exp1_pareto_humaneval.py`
  - Cost-performance Pareto plot on MBPP (default cost proxy: average generation time).
- `exp2_pool_scalability_humaneval.py`
  - Pool size scalability sweep on MBPP with stratified random sub-pool construction (builder/planner/checker/refactor).
- `exp3_task_oriented_teaming_humaneval.py`
  - Team size distribution and task-family-oriented agent/domain usage on MBPP.
- `exp3_task_oriented_teaming_humaneval_multi.py`
  - Multi-report aggregation for team-size/family/domain comparison across MBPP runs.
- `exp3_cross_dataset_teaming.py`
  - Cross-dataset teaming comparison (Team Size + selected Agent category/domain + selected role distribution).
- `exp4_proxy_reward_correlation_humaneval.py`
  - Correlation between MCTS proxy reward (`avg_reward`) and true task success on MBPP.

## Quick Start

Run from the repository root:

```powershell
python src/analysis/exp1_pareto_humaneval.py --local_model_dir <YOUR_MODEL_DIR>
python src/analysis/exp2_pool_scalability_humaneval.py --local_model_dir <YOUR_MODEL_DIR>
python src/analysis/exp3_task_oriented_teaming_humaneval.py --run_eval --local_model_dir <YOUR_MODEL_DIR>
python src/analysis/exp3_task_oriented_teaming_humaneval_multi.py --report_glob "runs/mbpp_report*.json"
python src/analysis/exp3_cross_dataset_teaming.py --dataset_specs "MBPP|runs/mbpp_report.json" "MATH|runs/math_report.json"
python src/analysis/exp4_proxy_reward_correlation_humaneval.py --run_eval --local_model_dir <YOUR_MODEL_DIR>
```

By default these scripts now use `data/datas/mbpp_test.jsonl` and call `evaluation.eval_mbpp`.

Outputs are written to:

- `src/analysis/outputs/exp1_pareto`
- `src/analysis/outputs/exp2_scalability`
- `src/analysis/outputs/exp3_teaming`
- `src/analysis/outputs/exp3_teaming_multi`
- `src/analysis/outputs/exp3_cross_dataset`
- `src/analysis/outputs/exp4_reward_corr`

Each output directory contains:

- machine-readable summaries (`*.json`)
- table files (`*.csv`)
- ready-to-use figures (`*.png`)

## Optional Analysis Metadata in Report

Two optional flags were added to `evaluation.eval_humaneval`:

- `--analysis_include_selection_trace`
- `--analysis_include_run_meta`

They are opt-in only (default off), so existing tests/flows are unchanged.
When running through `evaluation.eval_mbpp`, these flags are forwarded to `eval_humaneval`.
