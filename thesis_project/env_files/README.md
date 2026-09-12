# Environment Files

This folder contains exported conda environments used to produce the results 
and figures in this thesis. Environments were exported directly from Snellius 
(SURF's national supercomputer) where all experiments were run.

## Environments

| File | Used for |
|---|---|
| `vlm_qwen_reproducing_env.yml` | Qwen2-VL experiment jobs (SLURM `.job` scripts under `reproducing_code/jobs/`) |
| `vlm_qwen_reproducing_env_minimal.yml` | Same as above, listing only top-level installed packages (see note below) |
| `vlm_qwen3_reproducing_env.yml` | Qwen3-VL experiment jobs |
| `vlm_reproducing_env_plot.yml` | Figure generation and result visualization (used to run `Emma_figures_and_results_processing.ipynb` and `view_moments_language_vision_results.ipynb`) |

## Recreating an environment

```bash
conda env create -f vlm_qwen_reproducing_env.yml
```

This will create an environment with the same name as specified in the file 
(e.g. `vlm_qwen_reproducing_env`). To use a different name:

```bash
conda env create -f vlm_qwen_reproducing_env.yml -n my_custom_name
```

## A note on portability

The full `.yml` exports (`conda env export`) pin exact package builds and 
dependency versions specific to the Snellius cluster's OS and CUDA setup. 
This gives an exact reproduction on similar systems, but may fail to install 
cleanly on a different OS or hardware configuration.

`vlm_qwen_reproducing_env_minimal.yml` was generated with 
`conda env export --from-history`, which lists only the packages explicitly 
installed (letting conda re-resolve dependency versions for the target 
system) — this is more portable but less exact. For `vlm_qwen3_reproducing_env` 
and `vlm_reproducing_env_plot`, this flag did not meaningfully reduce the 
package list (likely because those environments were originally created from 
a single batch install rather than many incremental installs), so no separate 
minimal file is provided for those two.

If the full `.yml` fails to install on your system, try removing the 
build-hash suffixes (e.g. `=py3.11_cuda12.1_cudnn9.1.0_0`) from each line and 
letting conda resolve compatible versions instead.

