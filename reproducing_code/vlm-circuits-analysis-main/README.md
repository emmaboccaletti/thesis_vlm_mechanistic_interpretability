# Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs

This repository contains the code for the experiments of our VLM circuit overlap [paper](https://arxiv.org/abs/2506.09047) and the [project website](https://technion-cs-nlp.github.io/vlm-circuits-analysis).


## Repository structure
* data: Contains generated prompt csv files for each model and task. These files can also be generated automatically by the scripts and code. Each task directory also contains an images.tar.gz file that hosts all images on git-lfs.
* data\_generation: Contains scripts to generate the images (that are found in the data folder).
* third\_party: Contains a fork of the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library with edits we made to make it support VLMs (specifically the VLMs we analyzed as well as Llama3.2).
* All script files (`script_.*.py`) contain the code for running the experiments described in the paper as GPU jobs.
* Other files contain processes used in experiments (attribution patching, faithfulness evaluations, alignment between modalities, circuit overlap measurements, utility functions for each task, etc).
* `docs` contains code for the project website.

## MOMENTS utilities
The MOMENTS pipeline now includes a small checker for token-length drop rates:

```bash
python3 check_moments_drop_rates.py --max_images 1
```

By default it reports the drop rate for `random_pair`, `language_only`, and `both` using the Qwen processor/tokenizer path, without loading model weights.
