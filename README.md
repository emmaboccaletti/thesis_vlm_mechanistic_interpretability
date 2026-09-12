# Mechanistic Interpretability of Vision-Language Models

Master's thesis project studying how vision-language models (VLMs) internally represent and process visual vs. textual information, using mechanistic interpretability tools (attribution patching, circuit discovery, activation patching). All experiments are run on [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer) (SURF's national supercomputer).

The project has two parts:

1. **Reproducing** the circuit-discovery methodology from [*Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs*](https://arxiv.org/abs/2506.09047) (Technion CS-NLP) across several reasoning tasks (arithmetic, counting, color ordering, factual recall, sentiment analysis) on Qwen2-VL and Gemma 3.
2. **Applying that methodology to a new task**: detecting whether a football clip depicts a goal, and distinguishing goal events from other event types (corner/throw-in, shot-on-target), using the MOMENTS football-highlights dataset (video, audio commentary, and transcripts aligned to SoccerNet/SoccerReplay-1988 broadcasts), to study modality-specific circuits on data outside the original paper's task set.

## Repository structure

```
models/            Scripts and SLURM jobs to download/smoke-test the VLMs under study
reproducing_code/  Fork of the original paper's codebase (vlm-circuits-analysis) + SLURM jobs
                    used to reproduce and extend its experiments
thesis_project/    Thesis-specific work: MOMENTS dataset processing, conda environments,
                    and the SLURM jobs that build the MOMENTS-derived experiment data
```

### `models/`
- `download_model.py` — downloads a Hugging Face model repo (e.g. `Qwen/Qwen2-VL-7B-Instruct`) to local storage.
- `jobs/` — SLURM job scripts to download and smoke-test each model (Qwen2-VL, Qwen3-VL, Gemma 3, Pixtral) on the cluster.
- `job_outputs/` — captured stdout logs from those jobs.
- `qwen2vl7b/`, `qwen3vl8b/` — local model weight directories (ignored by git; only `.gitkeep` placeholders are tracked).

### `reproducing_code/`
- `vlm-circuits-analysis/` — a copy of the [original paper's repository](https://github.com/technion-cs-nlp/vlm-circuits-analysis), the forked version I worked on can be found [here](https://github.com/emmaboccaletti/vlm-circuits-analysis.git). Contains the circuit discovery (`script_node_circuit_discovery_and_eval.py`), cross-modality analysis, node-intersection, and backpatching experiment scripts, plus per-task utilities and prompt-generation code. See its own [README](reproducing_code/vlm-circuits-analysis/README.md) and [MOMENTS_overview.md](reproducing_code/vlm-circuits-analysis/MOMENTS_overview.md) for a file-by-file walkthrough of how the code maps to the paper's methodology.
- `jobs/` — SLURM jobs that run the above scripts per task/model on the cluster's H100 GPUs.
- `job_outputs/` — captured stdout/stderr logs from those runs.

### `thesis_project/`
- `data/` — dataset access notes; large datasets (e.g. MOMENTS) live on shared scratch storage and are only symlinked here, never committed. See [`data/README.md`](thesis_project/data/README.md) and [`data/moments-dataset-notes.md`](thesis_project/data/moments-dataset-notes.md) for the dataset layout and key preprocessing decisions.
- `data_generation/` — scripts that turn the raw MOMENTS clips into prompt datasets usable by the circuit-discovery pipeline (frame extraction, vocabulary building, category annotation, dataset assembly).
- `env_files/` — exported conda environments used to reproduce results (see [`env_files/README.md`](thesis_project/env_files/README.md) for which environment to use for what).
- `jobs/` — SLURM jobs for MOMENTS frame extraction and dataset generation.

## Getting started

1. Create the relevant conda environment (see [`thesis_project/env_files/README.md`](thesis_project/env_files/README.md)):
   ```bash
   conda env create -f thesis_project/env_files/vlm_qwen_reproducing_env.yml
   ```
2. Download the model(s) you need:
   ```bash
   python models/download_model.py Qwen/Qwen2-VL-7B-Instruct
   ```
3. Generate or symlink the required datasets under `thesis_project/data/` and/or `reproducing_code/vlm-circuits-analysis/data/`.
4. Submit the relevant SLURM job from `reproducing_code/jobs/` or `thesis_project/jobs/`, or run the underlying Python script directly for local debugging.

## Data (Zenodo)

The data collected and produced for this thesis — circuit analysis in vision-language
models (VLMs), including the MOMENTS (football broadcast) goal-detection experiments —
is archived on [Zenodo](https://zenodo.org/records/22727121?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlkZDY2MWRiLWQ0YzktNGViMS05ZjUzLTM0ZTYwNWFjZWZhZSIsImRhdGEiOnt9LCJyYW5kb20iOiJkMWZlYjQzNjVlYzIwZTRjYzM4MWFmYmI4ZmQ4YmQ1MiJ9.GmO04ve70NR2YFYwDg2PDbDjDA-v7A-vOjhmdGZKlRHhpI1g_r5-auVwlA_V4Lbm01mKa5lkg9y_sCUziH3i8Q). It has two top-level folders:

### `thesis_project/` (~841 MB) — raw MOMENTS dataset construction

The source data built from football-match broadcast videos, before any model/circuit
analysis:

- **`data/MOMENTS_categories/`** — CSV/JSON annotation files that classify short video
  clips ("important moments," e.g. `IM_14`) from the MOMENTS video corpus:
  - `candidates_events.csv` — clips auto-matched to event categories (goal, foul, etc.)
    via keyword regex matching against commentary transcripts, with a similarity score
    and the matched keyword/phrase.
  - `category_annotation.csv` / `category_annotations.json` — the curated/annotated
    event-type labels per clip (e.g. `GOAL`) grouped by category.
  - `goals_annotation.csv` — the subset of clips specifically annotated as goal events.
  - Each row references a `clip_id` (an 8-char video ID), `group_idx`, `clip_name`
    (`IM_n`), and paths to the original `.mp4`/`.json` files.

    **Note:** these paths are absolute paths from the original machine
    (`/home/eboccaletti/thesis_project/...`) and will not resolve for anyone else —
    they're left over from data collection. They're not sensitive, just stale; only
    the `clip_id` / `group_idx` / `clip_name` / label columns are meaningful if you're
    reproducing anything from this data.

- **`data/MOMENTS_frames/frames/`** — 1,429 PNG frames extracted from those clips,
  organized as `frames/<clip_id>/im/<group_idx>/<IM_n>/...` for "important moment"
  frames and `.../nim/<group_idx>/<NIM_n>/...` for "non-important moment"
  (negative/control) frames, across 62 distinct match clips. This is the image data
  behind the goal-detection task used later in the circuit analysis.

### `reproducing_code/vlm-circuits-analysis/` (~19 GB) — circuit-analysis pipeline data/results/figures

This folder holds the `data/` and `figures/` inputs/outputs of the `vlm-circuits-analysis`
codebase — **it does not contain the analysis source code itself** (no `.py` files).
Anyone wanting to rerun the pipeline will need that codebase separately; this is the
data it consumed and produced.

- **`data/<task>/`** for six tasks — **arithmetic, color_ordering, counting,
  factual_recall, goal, sentiment_analysis**:
  - Input stimuli: images (`images/`, or CLEVR scenes for `color_ordering` — see its
    `README.md`, which notes the CLEVR dataset must be downloaded separately) and
    `*_textual_data.csv` / `*_visual_data.csv` prompt sets per model.
  - `results/<model>/` (gemma-3-12b-it, qwen2-7b-vl-instruct, pixtral-12b) — the
    original authors' circuit-discovery outputs: `.pt` tensors for node attribution
    scores (`node_scores/nap_ig_*_metric=LD.pt`), faithfulness circuits
    (`faithfulness_LD_*_node_circuit.pt`), cross-modal interchange results, and
    backpatching results.
  - `results_emma/<model>/` — my own reruns of the same pipeline (for
    comparison/validation against the original results).
- **`data/goal/`** — composite images built from the goal-detection frames, plus
  **noised versions** at several noise levels (`composites_noisy_sigma0p05` …
  `sigma0p5`) used for robustness/faithfulness testing.
- **`data/moments_goal/`** — the core MOMENTS experiment tying it together: CSVs
  pairing "goal" vs. control clips in various conditions (`both_data*`,
  `language_only_data*`, `vision_only_data*`, `goal_vs_corner_data`,
  `random_pair_data*`, with/without noise sigma, with/without POS control), and
  `results/` per model/config (qwen2-7b-vl-instruct, qwen3-8b/vl variants) containing
  behavioral-evaluation CSVs (`behavior_clean*.csv`) and the same kind of
  `.pt` circuit/faithfulness/backpatching/intersection tensors, including
  `qwen2_behavioral_evaluation/` (raw model behavior on clean clips).
- **`figures/`** — the final plots/tables: PDF/PNG faithfulness and intersection
  comparisons, `appendix_heatmaps/` (per-task, per-model heatmaps),
  `language_only_results/`, `moments_circuit_comparison/` (CSV summaries:
  faithfulness, layer composition, ranked hook scores, overlap-vs-random-baseline),
  and `moments_language_vision_results/` broken into `backpatching/`, `behavior/`,
  `cross_modality/`, `intersection/`, `new_experiment_variants/`.
- Two notebooks: `Emma_figures_and_results_processing.ipynb` (turns the `.pt`/CSV
  results into the figures above) and `view_moments_language_vision_results.ipynb`
  (inspects/visualizes the MOMENTS goal-circuit results across language vs. vision
  modalities).

## Reference

Nikankin et al., *Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs*. [arXiv:2506.09047](https://arxiv.org/abs/2506.09047). [Project website](https://technion-cs-nlp.github.io/vlm-circuits-analysis).
