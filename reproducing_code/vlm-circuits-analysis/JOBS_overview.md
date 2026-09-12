# Jobs overview: how the scripts were actually run on Snellius

`overview.md` documents the original paper's methodology; `MOMENTS_overview.md` documents what the extended codebase in this directory does. This file documents a different thing: **how** the scripts were actually invoked as SLURM batch jobs on Snellius — which script, with what CLI arguments and conditions (model, task, counterfactual mode, sigma noise level, IG steps, circuit-size percentages), and what role each job plays in the overall pipeline.

Source directories:
- `reproducing_code/jobs/*.job` — the original 5-task jobs (arithmetic, counting, color_ordering, factual_recall, sentiment_analysis).
- `reproducing_code/jobs/MOMENTS/*.job`/`.sh` — the MOMENTS extension jobs (all `moments_goal`).
- `reproducing_code/job_outputs/*.out`/`.err` — captured logs, but **only for the original 5-task jobs**. There is no `job_outputs/MOMENTS/` subfolder in this repo checkout, even though every MOMENTS job's `--output`/`--error` SBATCH directives point at `/home/eboccaletti/reproducing_code/job_outputs/MOMENTS/...`. So for the 5-task jobs below, claims about what actually ran/succeeded are cross-checked against real logs; for MOMENTS jobs, this document describes **what was launched** (from the job scripts' own commands), not confirmed outcomes — the logs simply aren't in this copy of the repo.

A related, useful fact for reading the MOMENTS jobs below: several of them invoke helper paths at `/home/eboccaletti/reproducing_code/jobs/run_node_cross_qwen_moments_goal_circuit_size_sweep_single.sh` and the `submit_qwen_moments_goal_goal_vs_corner_pipeline.job` orchestrator submits child jobs from `/home/eboccaletti/reproducing_code/jobs/...` (not `.../jobs/MOMENTS/...`). Since those referenced files currently live in this repo under `jobs/MOMENTS/`, these jobs were evidently written when the MOMENTS scripts still lived flat in `reproducing_code/jobs/`, before being reorganized into the `MOMENTS/` subfolder — the paths would need updating to this layout before resubmitting these particular jobs.

---

## Part 1 — the original 5-task jobs (`reproducing_code/jobs/*.job`)

Every job in this set follows the same skeleton: `module load 2025` + `Anaconda3/2025.06-1`, `source activate vlm_qwen_reproducing_env`, `cd .../vlm-circuits-analysis`, print a `nvidia-smi` snapshot, run one `python script_*.py` command, print start/end timestamps. What varies is the script, model, task, and a couple of numeric arguments.

### `script_node_circuit_discovery_and_eval.py` — circuit discovery
Run once per (model, task) pair, and again with `--language_only` for the language-only ("L") baseline. Conditions:

| Model | `--model_path` | Tasks covered | `--ap_ig_steps` | Notes |
|---|---|---|---|---|
| `qwen2-7b-vl-instruct` | `/home/eboccaletti/models/qwen2vl7b` | arithmetic, counting, color_ordering, factual_recall, sentiment_analysis (VL + L each) | 5 | `gpus=1`, `time=24:00:00` |
| `gemma-3-12b-it` | `/home/eboccaletti/models/gemma3_12b_it` | same 5 tasks (VL + L each) | 5 | `gpus=4` — Gemma 3 needed 4 H100s where Qwen2 needed 1; job output confirms `Detected 4 GPUs` |
| `pixtral-12b` | `/home/eboccaletti/models/pixtral-12b` | counting only | 1 | `gpus=1`, `time=00:30:00` |

Confirmed from `job_outputs/`: Qwen2 and Gemma runs completed (`End time: ...` printed, model loaded, no traceback) for arithmetic, counting, factual_recall, sentiment_analysis, color_ordering, in both VL and L variants.

### `script_node_cross_modality_analysis.py` — cross-modal interchange
One job per task, Qwen2 only (`run_node_cross_qwen_{arithmetic,counting,color_ordering,factual_recall,sentiment_analysis}.job`), all identical in shape:
```
python script_node_cross_modality_analysis.py \
  --model_name qwen2-7b-vl-instruct --model_path .../qwen2vl7b \
  --task_name <task> --l_circuit_percentage 0.01 --vl_circuit_percentage 0.03
```
`gpus=1`, `time=12:00:00`. Logs confirm each completed (loads both the textual and visual per-task CSVs, prints `End time`).

### `script_node_intersection.py` — node intersection
Same shape as cross-modality, one job per task, Qwen2 only, `l_circuit_percentage`/`vl_circuit_percentage` hard-coded in the file (`time=03:00:00`). Notably, for at least `arithmetic` there are **two** archived output IDs (`21869006` and `21875158`) with *different* percentage pairs logged (0.05/0.08 vs 0.01/0.03) — meaning the percentages in the `.job` file were hand-edited and the job resubmitted between runs, rather than the sweep being parameterized. The `.err` logs contain the actual printed result dict (head/MLP IoU and baseline IoU, overall and split by Data/Query/Generation position group, with and without position information) — e.g. for arithmetic at 5%/8%: `vl_head_iou: 0.23`, `l_head_iou: 0.37`, etc. So for the original tasks, per-task intersection numbers exist only for whichever one or two percentage pairs happened to be run this way — there's no systematic percentage sweep for the 5 original tasks (unlike the MOMENTS jobs, which do sweep systematically — see Part 2).

### `script_backpatching_experiment.py` — layer-window backpatching
One job per task, Qwen2 only, `time=24:00:00`, `--src_layer_range`/`--dst_layer_range` set per task to match the paper's reported best backpatching window (each `.job` file has a trailing comment quoting the paper's "best setting" for that model/task, e.g. arithmetic: `11 → 7 (3)`, counting: `20 → 15 (3)`). The arithmetic run completed and printed `(True, 3) tensor(0.7700)` — a (success flag, window size) / accuracy-style result.

`run_backpatch_vqav2_qwen.job` runs the same script with no `--task_name` (so the default task applies) against layer range `15 28`→`0 16`, labeled as a "counting" VQAv2 backpatch job — this is the entry point that exercises `script_backpatch_vqav2.py`'s dataset loader indirectly through the shared backpatching CLI (see `MOMENTS_overview.md`'s note on `script_backpatch_vqav2.py`).

### `attempt_fix_OOM_script_backpatching_experiment.py` — color_ordering variant
`attempt_fix_OOM_run_backpatching_exp_qwen_color_ordering.job` runs the same `color_ordering` task and layer range (`13 26` → `0 16`) as `run_backpatching_exp_qwen_color_ordering.job`, but calls `attempt_fix_OOM_script_backpatching_experiment.py` instead (which moves cached activations to CPU and pulls back only the needed layer slice per hook — see `MOMENTS_overview.md`), with `--mem` raised to **360G** and `--cpus-per-task` raised to 16.

### Data-generation jobs (not analysis, but included here since they're in the same `jobs/` folder)
- `generate_counting_images.job` — `gpu_a100`, runs `data_generation/generate_counting_images.py --prompt_template_idx 0 --seeds 42 43 44 45 46`.
- `label_counting_images.job` — `gpu_a100`, runs `data_generation/label_counting_images.py --input_dir ~/data/counting/raw_images/1 --output_dir .../data/counting/images --model gpt-4o` (needs `OPENAI_API_KEY` in the environment — the job script has commented-out instructions for setting it, not a real secret).
- `rename_counting_images.job` — `gpu_a100` (requested but not really used — script has no GPU code), runs `data_generation/rename_counting_images.py` with no arguments.

---

## Part 2 — the MOMENTS jobs (`reproducing_code/jobs/MOMENTS/*`)

All MOMENTS jobs target `--task_name moments_goal`. Dataset/vocabulary generation itself (`create_moments_dataset.py`, `build_moments_qwen_vocab.py`, frame extraction) is **not** in this folder — those jobs live in `thesis_project/jobs/` and are documented via `thesis_project/data/moments-dataset-notes.md`.

### Priority file A — `run_node_cross_qwen_moments_goal_circuit_size_sweep_single.sh` and its caller

`run_node_cross_qwen_moments_goal_circuit_size_sweep_single.sh` is a **reusable helper, not itself a SLURM job** (no `#SBATCH` lines). It takes 6 positional arguments — `LABEL MODEL_NAME MODEL_PATH CF_MODE DATA_FILE SCORES_DIR` — and for each of 4 (L%, VL%) circuit-size pairs (`0.10/0.10`, `0.20/0.30`, `0.30/0.50`, `0.50/0.70`) runs:
```bash
python script_node_cross_modality_analysis.py \
  --model_name "${MODEL_NAME}" --model_path "${MODEL_PATH}" \
  --task_name moments_goal --moments_cf_mode "${CF_MODE}" \
  --moments_data_file "${DATA_FILE}" --scores_dir "${SCORES_DIR}" \
  --output_file "${SCORES_DIR}/cross_modality/circuit_size_sweep/<tag>/faithfulness_nodes_cross_interchanges_LD_${SLURM_JOB_ID}.pt" \
  --l_circuit_percentage "${L_PCT}" --vl_circuit_percentage "${VL_PCT}"
```
This is used two ways:
1. **As a batch orchestrator**, via `run_node_cross_qwen_moments_goal_circuit_size_sweep.job`, a SLURM **array job** (`--array=0-7`) whose `VARIANTS` array picks a different (label, model, cf_mode, data file, scores dir, conda env) tuple per array index: array indices 0–2 are Qwen2 "baseline" runs (both/language_only/vision_only) reading from a **dated snapshot directory with a literal space in its path**, `data/moments_goal as of 7sep/`; indices 3–6 are Qwen2 "POS-aware" sigma variants (`sigma0p05`, `sigma0p1`, `sigma0p12`, `sigma0p5`, each reading `data/moments_goal/both_data_sigma{X}.csv`); index 7 is a Qwen3 "both" run (switching conda env to `vlm_qwen3_reproducing_env` and model path to `qwen3vl8b` via a shell ternary). Every variant reduces to the same 4-pair L/VL sweep via the helper script.
2. **Standalone, per variant**, via single (non-array) jobs like `run_node_cross_qwen_moments_goal_baseline_both_sweep.job` (baseline "both", reading the `as of 7sep` snapshot), `run_node_cross_qwen_moments_goal_baseline_language_sweep.job`, `run_node_cross_qwen_moments_goal_baseline_vision_sweep.job`, and the four `run_node_cross_qwen_moments_goal_pos_sigma{0p05,0p1,0p12,0p5}_sweep.job` files, plus Qwen3 equivalents (`run_node_cross_qwen3_moments_goal_baseline_both_sweep.job`) — each just calls the same `.sh` helper with one hard-coded set of positional args. These look like they exist so a single failed/incomplete array-index could be resubmitted on its own without resubmitting the whole array.

"POS-aware sigma" here refers to the Gaussian-noise vision-corruption sweep described in `thesis_project/data/moments-dataset-notes.md` (`create_moments_dataset.py --noise_sigma`/`--extra_noise_sigmas`) — "POS" flags that these particular CSVs were built with the POS-bucketed same-token-length language-corruption vocabulary (as opposed to the `_withoutPOS` CSV variants also present in `data/moments_goal/`).

### Priority file B — `run_node_intersection_qwen_moments_goal_both_pos_sigma0p5_table5_sweep.job` and its family

This is a single, self-contained (non-array) SBATCH job: `gpu_h100`, 1 GPU, 9 CPUs, 80G mem, 8h walltime. It sets:
```
SCORES_DIR=data/moments_goal/results/qwen2-7b-vl-instruct_both_POS_sigma0p5_nap_ig=5
DATA_FILE=./data/moments_goal/both_data_sigma0p5.csv
```
and loops over **8** (L%, VL%) pairs — a finer, wider grid than the cross-modality sweep above: `0.01/0.03`, `0.02/0.05`, `0.05/0.08`, `0.10/0.10`, `0.20/0.30`, `0.30/0.50`, `0.50/0.70`, `0.70/0.90` — calling `script_node_intersection.py` (not cross-modality) for each pair, writing results under `${SCORES_DIR}/intersection/table5_sweep/<tag>/`.

"table5" most plausibly refers to a specific results table in the thesis write-up that reports intersection/overlap numbers across this exact 8-point circuit-size grid — the same 8 pairs recur verbatim across every `*_table5_sweep.job` file. Conceptually this differs from the cross-modality sweep in Priority file A: `script_node_intersection.py` measures **which components are shared** between the L and VL circuits at a given size (IoU-style overlap, with/without position info, split by Data/Query/Generation — see Part 1's intersection section), while `script_node_cross_modality_analysis.py` measures **causal interchange faithfulness** (swapping activations between modalities and checking whether behavior transfers) — different questions, same size-sweep grid reused for both.

The `..._table5_sweep.job` family (read and confirmed to follow the identical 8-pair-grid template, differing only in `SCORES_DIR`/`DATA_FILE`/optional `--moments_total_prompt_count`):
- `run_node_intersection_qwen_moments_goal_both_table5_sweep.job` (no sigma)
- `run_node_intersection_qwen_moments_goal_both_pos_sigma{0p05,0p1,0p12,0p5}_table5_sweep.job` (4 sigma variants)
- `run_node_intersection_qwen_moments_goal_goal_vs_corner_table5_sweep.job` (adds `--moments_data_file data/moments_goal/goal_vs_corner_data.csv --moments_total_prompt_count 96`; carries `#SBATCH --dependency=afterok:26552096`, a **hard-coded historical job ID** — this file was edited and resubmitted by hand after manually confirming a specific prior circuit-discovery job had finished, rather than being a generically reusable template)
- `run_node_intersection_qwen_moments_goal_language_only_table5_sweep.job`, `..._vision_only_table5_sweep.job`
- `run_node_intersection_qwen3_moments_goal_both_pos_sigma0p12_table5_sweep.job` (Qwen3 equivalent)

Separately, `run_node_intersection_qwen_moments_goal_grid_1_20.job` is a **different, broader sweep**: a full 20×20 grid (`seq 1 20` for both L% and VL%, i.e. every 1–20% pair, 400 combinations), reading its counterfactual mode from an environment variable (`MOMENTS_CF_MODE`, default `both`) rather than being hard-coded — described in its own header comment as "the broadest retained-fraction search in the current thesis plan." This is a finer, low-percentage-focused grid compared to `table5_sweep`'s 8 hand-picked points spanning up to 90%.

### Behavioral evaluation jobs (`evaluate_moments_*_behavior*.job`)
These run inference once per prompt and score the model's raw next-token logits — no circuit/attribution analysis, no counterfactual pair needed beyond optional alignment filtering.

- `evaluate_moments_goal_behavior.job` / `evaluate_moments_goal_behavior_qwen2_sigma{0p05,0p1,0p5}.job` — `evaluate_moments_behavior.py` on Qwen2, scoring constrained yes/no goal-detection accuracy on `vision_only_data.csv` (base job) or `both_data_sigma{X}.csv` (sigma variants), writing to `data/moments_goal/results/qwen2.../behavior_clean*.csv`.
- `evaluate_moments_goal_behavior_qwen3.job` — same script, Qwen3 (`vlm_qwen3_reproducing_env`, `qwen3vl8b`, `--torch_dtype bfloat16`), on `both_data.csv`.
- `evaluate_moments_goal_behavior_qwen3_language_only.job` — identical to the above plus `--language_only`, explicitly to compare the text-only inference path against the standard multimodal path on the same CSV.
- `evaluate_moments_event_type_behavior.job` (Qwen2) / `evaluate_moments_event_type_behavior_qwen3.job` (Qwen3) — `evaluate_moments_event_type_behavior.py`, the 3-way goal/corner/shot behavioral evaluation, both reading `vision_only_data.csv` (the question is rewritten from goal-detection to 3-way at eval time, per `MOMENTS_overview.md`).
- `evaluate_moments_event_type_pairwise_behavior_qwen2.job` / `..._qwen3.job` — `evaluate_moments_event_type_pairwise_behavior.py`, same source CSV, scoring the three pairwise splits (goal-vs-corner, goal-vs-shot, corner-vs-shot).

All of these are short (`time` between 10 min and 1 h) single-GPU jobs — cheap compared to the circuit-discovery/backpatching jobs, since they're a single forward pass per prompt rather than clean+counterfactual+IG-interpolation forward/backward passes.

### `inspect_moments_faithfulness_baselines*.job`
- `inspect_moments_faithfulness_baselines.job` — runs `inspect_moments_faithfulness_baselines.py --model_path .../qwen2vl7b --percentages 0.3 0.5 0.9`, the three circuit sizes named in the script's own default.
- `inspect_moments_faithfulness_baselines_03.job` / `_05.job` / `_09.job` — presumably (filenames strongly imply, matching the `--percentages` value) the same call split into three separate single-percentage jobs — **not opened individually since the pattern is unambiguous from the base job and the `_array.job` variant below**, but flagged here rather than silently assumed.
- `inspect_moments_faithfulness_baselines_array.job` — a SLURM **array job** (`--array=0-24`, so 25 tasks), passing the same `--percentages 0.3 0.5 0.9` but adding `--eval_prompt_index "${SLURM_ARRAY_TASK_ID}"` and a per-index `--output_csv .../language_only_ld_baselines_eval_${SLURM_ARRAY_TASK_ID}.csv` — i.e. one array task per individual evaluation-set example (25 of them), presumably to parallelize or to isolate which specific MOMENTS clips produce baseline anomalies (this is exactly the per-sample tracing use case `inspect_moments_faithfulness_baselines.py` was built for, per `MOMENTS_overview.md`).

### Utility / post-processing jobs (CPU-only, `rome` partition — no GPU)
- `check_moments_drop_rates.job` — `check_moments_drop_rates.py --max_images 1`, the token-length drop-rate checker; 20 min, no GPU.
- `analyze_moments_circuit_comparison.job` — `analyze_moments_circuit_comparison.py --results_root data/moments_goal/results --output_dir figures/moments_circuit_comparison --top_k 5 10 20`; 30 min, no GPU (post-hoc analysis of already-saved node scores, as described in `MOMENTS_overview.md`).

### `run_node_circuit_qwen*_moments_goal_*.job` — circuit discovery
The `_L_ig5`/`_VL_ig5` filename suffixes mean exactly what they look like: `_L_ig5` passes `--language_only` (computing/saving node scores for the language-only view), `_VL_ig5` omits it (computing scores for the full vision-language view) — both write to the **same** `--output_dir` for a given condition (e.g. both `..._both_L_ig5.job` and `..._both_VL_ig5.job` target `data/moments_goal/results/qwen2-7b-vl-instruct_both_nap_ig=5`), since `script_node_circuit_discovery_and_eval.py` saves L and VL scores as separately-named files inside one directory. Both carry a header comment warning to "archive any existing active {l,vl} result files before submitting this job" — i.e. these are meant to be run in a controlled order, not resubmitted freely, since a second run would overwrite the first's output in the same directory.

Conditions covered (all Qwen2 unless noted), one `_L_ig5`/`_VL_ig5` pair each, `--ap_ig_steps 5`:
| Condition | `--moments_cf_mode` | Extra flags |
|---|---|---|
| `both` | `both` | — |
| `both_pos_sigma{0p05,0p1,0p12,0p5}` | `both` | `--moments_data_file .../both_data_sigma{X}.csv` |
| `goal_vs_corner` | `both` | `--moments_data_file .../goal_vs_corner_data.csv --moments_total_prompt_count 96` |
| `vision_only` | `vision_only` | `_L_ig5` and `_VL_ig5`, plus a `_smoketest` variant |
| `language_only` | `language_only` | `_L_ig5` and `_VL_ig5`, plus a `_smoketest` variant |

Qwen3 equivalents exist for `both` only: `run_node_circuit_qwen3_moments_goal_both_{L,VL}_ig5.job` (same flags, `vlm_qwen3_reproducing_env`, `qwen3vl8b`).

Smoketest jobs (`_smoketest` suffix) all use `--ap_ig_steps 1 --moments_max_images 1 --moments_total_prompt_count 2` — a 2-prompt, 1-frame, 1-IG-step configuration meant to validate the pipeline runs end-to-end cheaply before committing to a full (5-step, full-prompt-count) job. Three exist: `vision_only_smoketest`, `language_only_smoketest`, and `random_pair_smoketest` (the only job anywhere using `--moments_cf_mode random_pair`, i.e. an unpaired random counterfactual rather than a matched perturbation).

### `run_node_cross_qwen*_moments_goal_*.job` and `run_node_intersection_qwen*_moments_goal_*.job` — non-sweep, per-condition jobs
Beyond the sweep jobs covered above, each also has plain single-percentage-pair jobs mirroring the circuit-discovery conditions: `both`, `both_pos_sigma{0p05,0p1,0p12,0p5}`, `goal_vs_corner`, `language_only`, `vision_only` (cross and intersection each), all fixed at `--l_circuit_percentage 0.01 --vl_circuit_percentage 0.03` (the same pair used for the original 5 tasks' cross/intersection jobs), reading scores from the matching `results/qwen2-7b-vl-instruct_<condition>_nap_ig=5` directory built by the circuit-discovery jobs above. `run_node_intersection_qwen_moments_goal_vision_only_smoketest.job` is the one smoketest-labeled variant in this pair of scripts, but its command is identical in shape to the plain `vision_only` job (0.01/0.03, no `--moments_max_images`/`--moments_total_prompt_count` caps) — read in full to check this, and confirmed it isn't actually a cheaper smoketest configuration despite the name, just a normal run against the `vision_only` results directory.

### `run_backpatching_exp_qwen*_moments_goal_*.job` — backpatching
Same layer range as the original tasks' Qwen2 arithmetic job (`--src_layer_range 6 19 --dst_layer_range 0 13`) across `both`, `both_pos_sigma{0p05,0p1,0p12,0p5}`, `goal_vs_corner`, `language_only`, `vision_only` conditions, each writing to a mode-specific `.../backpatching` subdirectory of its circuit-discovery results directory. `run_backpatching_exp_qwen_moments_goal_vision_only_smoketest.job` drops the `--output_dir` override (writing to the script's default location instead) — the only difference from the plain `vision_only` job; like the intersection smoketest above, it is not obviously cheaper (same layer range, no prompt-count cap), so "smoketest" here likely just means "first trial run, not yet pointed at the final output location," not a reduced-cost configuration. `run_backpatching_exp_qwen3_moments_goal_both_pos_sigma0p12.job` is the Qwen3 equivalent for that one condition.

### `submit_qwen_moments_goal_goal_vs_corner_pipeline.job` — the orchestrator
A lightweight (1 CPU, 1G mem, 5 min) wrapper that doesn't run any analysis itself — it calls `sbatch --parsable` three then two more times to chain a pipeline:
1. Submits the `goal_vs_corner` L and VL circuit-discovery jobs (`run_node_circuit_qwen_moments_goal_goal_vs_corner_{L,VL}_ig5.job`) with no dependency.
2. Captures both job IDs, builds `dependency="afterok:${language_job}:${vision_language_job}"`.
3. Submits the `goal_vs_corner` intersection, cross-modal, and backpatching jobs all with that same dependency string, so they only start once **both** circuit jobs have finished successfully.

This is the one place in the MOMENTS jobs that expresses an explicit dependency graph via SLURM's `--dependency` mechanism rather than being run by hand in sequence; the `..._table5_sweep`/`..._circuit_size_sweep` jobs for `goal_vs_corner` instead hard-code a specific completed job ID (`afterok:26552096`) rather than being launched by this orchestrator, meaning they were added/resubmitted later, by hand, after the orchestrator's original circuit jobs had already completed.

As noted at the top of this document, this orchestrator's `JOB_DIR` points at `/home/eboccaletti/reproducing_code/jobs` (flat), not `.../jobs/MOMENTS` — so it needs an updated path (or the referenced `.job` files copied back to the flat directory) to run as checked into this repo today.

---

## Decoding a job filename not covered above

- **`smoketest`**: a cheap validation run before a full one. For circuit-discovery jobs this reliably means `--ap_ig_steps 1 --moments_max_images 1 --moments_total_prompt_count 2`; for the one intersection/backpatching job carrying the name, it turned out (checked directly) to just mean "not yet using the final output path," not a reduced-cost run — don't assume the flag combination without checking.
- **`sigma0pXX`**: the Gaussian-noise standard deviation used for MOMENTS' vision-corruption sweep (`0.05`, `0.1`, `0.12`, `0.2`, `0.5`), always paired with a `both_data_sigma{X}.csv`/`language_only_data_sigma{X}.csv`/etc. data file.
- **`POS`** (as in `both_POS_sigma0p5`): the dataset was built using the POS-bucketed same-token-length language-corruption vocabulary (the default), as opposed to the `_withoutPOS` CSV variants also present in `data/moments_goal/`.
- **`_L_ig5` / `_VL_ig5`**: `_L` passes `--language_only` (saves the language-only node scores); `_VL` omits it (saves the vision-language node scores) — both write into the same result directory for a condition.
- **`table5_sweep`**: the fixed 8-point (L%, VL%) grid (`0.01/0.03` … `0.70/0.90`) used for `script_node_intersection.py`, matching a specific results table in the thesis.
- **`circuit_size_sweep`**: the coarser 4-point (L%, VL%) grid (`0.10/0.10` … `0.50/0.70`) used for `script_node_cross_modality_analysis.py`.
- **`grid_1_20`**: the full 20×20 (1–20%, 1–20%) intersection grid — a different, much finer sweep than `table5_sweep`.
- **Array jobs (`--array=...`)**: used for two purposes in these jobs — sweeping over a list of (model, condition) tuples (`run_node_cross_qwen_moments_goal_circuit_size_sweep.job`, 8 variants) or sweeping over individual evaluation examples (`inspect_moments_faithfulness_baselines_array.job`, 25 examples via `SLURM_ARRAY_TASK_ID`).
- **`baseline`** (as in `..._baseline_both_sweep.job`): refers to the pre-7-September Qwen2 runs reading from the dated `data/moments_goal as of 7sep/` snapshot directory, as opposed to the current `data/moments_goal/` CSVs.

## What I could not confidently characterize

- `inspect_moments_faithfulness_baselines_03.job` / `_05.job` / `_09.job` were not opened individually — their purpose (one job per `--percentages` value) is inferred from the filename pattern and the base/array jobs' content, not confirmed by reading each file.
- Whether any MOMENTS job actually completed successfully cannot be confirmed at all from this repo — there are no `job_outputs/MOMENTS/*.out`/`.err` files here.

## Difficulties encountered

- **Pixtral node-circuit discovery**: getting `script_node_circuit_discovery_and_eval.py` running on Pixtral took several attempts (`pixtral_12b_2409_node_circuit_21638368`, `pixtral_node_circuit_21638772`, `pixtral_node_circuit_21650656` in `job_outputs/`) before landing on the current `run_node_circuit_pixtral.job` configuration (`--ap_ig_steps 1`, `--task_name counting`).
- **Qwen2 color_ordering backpatching**: the plain `run_backpatching_exp_qwen_color_ordering.job` needed multiple resubmissions (job IDs `21879554`, `21891742`, `21893477` in `job_outputs/`) before `attempt_fix_OOM_run_backpatching_exp_qwen_color_ordering.job` was written as a higher-memory (360G, 16 CPUs) variant calling `attempt_fix_OOM_script_backpatching_experiment.py` instead. No output log for that fix job exists in this repo, so its outcome isn't recorded here.
