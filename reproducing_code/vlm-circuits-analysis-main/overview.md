# How these files fit together

### analysis_utils.py
analysis_utils.py and the task-specific loaders build the dataset described in Section 3: five analogous tasks, each with textual and visual variants, with prompts aligned into data/query/generation positions.

### vision_language_prompts.py
vision_language_prompts.py defines the prompt object that carries a clean prompt and its counterfactual, which is exactly what Section 2.2 and 2.3 require: each prompt p is paired with a counterfactual p' that yields a different answer.

This file defines the central data object used everywhere else.

#### class VLPrompt
The class VLPrompt "is a dataclass that represents a Vision-Language Prompt (a prompt that optionally contains a visual input). Also includes a counterfactual prompt and counterfactual images."
It stores:
- prompt: clean text prompt,
- images: associated clean image(s),
- answer: clean answer,
- cf_prompt: counterfactual prompt,
- cf_images: counterfactual images,
- cf_answer: counterfactual answer,
- metadata.

In Section 2.2, each:
- clean prompt p with answer r
- is paired with a counterfactual prompt p' with answer r'.
In Section 2.3, the evaluation intervention uses:
- counterfactual activations from p'
- while running p.
This class is how the repo packages those paired examples.

The class also defines equality, ordering, and hashing in a way that includes image contents. That matters because: they use set/deduplication operations, they shuffle/split prompts reproducibly, prompt identity must include both text and image.

#### vlp_collate_fn(batch)

`vlp_collate_fn(batch)` is a DataLoader collate function. It bundles a list of VLPrompt objects into batched lists of prompts/images/answers/counterfactuals. 
Created because rest of the code relies heavily on batched evaluation and patching.

In conceptual terms, this file is not “methodology”; it is the data container for the paper’s counterfactual prompt formalism.

### attr_patching.py
attr_patching.py implements the AP-IG scoring from Equation 1: estimate each component’s importance by combining activation differences with gradients along an interpolation path between counterfactual and clean inputs.

This is the most methodologically important file, because it implements the paper’s circuit discovery score.

The paper’s Equation 1 defines AP-IG, a component gets a score based on:
- the difference between counterfactual and clean activations,
- multiplied by gradients of the logit-difference objective,
- integrated along interpolated inputs between counterfactual and clean embeddings.

That is exactly what node_attribution_patching_ig(...) does.

#### node_attribution_patching_ig(...)

`node_attribution_patching_ig(...)` estimates importance scores for node-like components:
- MLP post activations,
- attention head outputs hook_z,
- cross-attention states when relevant.

The paper says their circuit components are either entire attention heads or individual MLP neurons at specific output positions. The hook names here correspond to the tensors from which those node scores are later derived.

The function loops over prompts and, for each one:

tokenizes the clean and counterfactual answers,
- runs a forward pass on the counterfactual prompt to cache counterfactual activations,
- runs a forward pass on the clean prompt to cache clean activations,
- computes diff_cache = cf_cache - clean_cache,
- runs integrated-gradient-style interpolated forwards from counterfactual input embeddings toward clean embeddings,
- backpropagates the chosen metric,
- accumulates (activation difference × gradient).

That is the code equivalent of the formula in Section 2.2. The interpolated hook on hook_embed is especially important: the paper defines integration over the path between e' and e, the counterfactual and clean input embeddings. The inner hook function literally constructs that interpolation.

The default metric="LD" means the gradient target is logit difference. That matches the paper exactly.

The final score is averaged across prompts, again matching the paper’s statement that component importance is averaged across prompts.

A subtle point: the function stores scores per hook tensor, not yet as final Component objects. Later code converts:

hook_z scores into attention-head components,
mlp.hook_post scores into neuron components.

So this file computes the raw importance landscape; general_utils.py turns that into a circuit.

#### node_attribution_patching(...)

This is the non-IG version: classic attribution patching without the integrated-gradient interpolation.

The paper’s main method is AP-IG, not plain AP, because AP-IG is a better first-order approximation and was motivated by prior faithfulness work. This function is therefore probably included for ablations or earlier experiments.

#### should_measure_hook(...)

A tiny filter that says which hook activations should receive attribution scores. It is just infrastructure.

THUS!
attr_patching.py is the implementation of Equation 1 and the “patching effects” underlying Figure 3 and Appendix D’s heatmaps. The paper says those patching effects are summed per position and layer to reveal different importance patterns, motivating the split into data/query/generation sub-circuits; this file is where those raw effects come from.


### general_utils.py
general_utils.py turns those raw scores into ranked circuit components: top heads and top MLP neurons at specific positions. That is the “construct a circuit by selecting a specific percentage of the highest-scoring components” step in Section 2.2.

### evaluation_utils.py
evaluation_utils.py implements Section 2.3: ablate all non-circuit components with counterfactual activations, compare to good and bad baselines, and normalize into faithfulness.

### script_node_circuit_discovery_and_eval.py
this script is the implementation of Sections 2.2 and 2.3 plus the experimental search over circuit sizes in Figure 4.

script_node_circuit_discovery_and_eval.py is the script that actually runs this whole discovery/evaluation pipeline over a model and task. Its outputs are what feed Figure 4 and the later circuit analyses. The paper says they use a 75/25 discovery/evaluation split and search over circuit sizes until they find a minimal circuit above 80% faithfulness; this script does exactly that.

`DISCOVERY_EVAL_SPLIT_PERCENT = 0.75` directly matches Appendix B.3: 75% discovery, 25% evaluation.

`METRICS = ["LD"]` means they use logit difference as the main metric, which matches the paper’s Equations 1 and 2, where component importance and faithfulness are defined in terms of the logit difference between the correct answer r and counterfactual answer r'

```
PERCENTAGES = sorted(
    torch.arange(0.0, 0.21, 0.01).tolist()  # High res in low amounts of nodes
    + torch.arange(0.3, 1.01, 0.2).tolist()  # Low res in high amounts of nodes
    + [0.001, 0.005]  # Some extra points
)
```
`PERCENTAGES` defines the grid of circuit sizes to test. The paper notes they search over percentages p ∈ {0.001, 0.005} ∪ {i·0.01} ∪ … and evaluate faithfulness across sizes, as shown in Figure 4. This list is the code version of that search grid.

#### analyze_faithfulness
`analyze_faithfulness`: Calculate the faithfulnes of each top-p% node-based circuit.

This function implements the “Next, we construct a circuit from the set of top-p (where p ∈ {0.001, 0.005} ∪ {i · 0.01 | i ∈ Z, 1 ≤ i ≤ 100}) percent of components and measure its faithfulness, repeating this process for different circuit sizes. Following earlier work (Nikankin et al., 2024; Ameisen et al., 2025), a circuit is considered sufficient for a task if it is the minimal circuit that achieves a faithfulness of over 80%."

It loads saved node scores from disk, takes their absolute value, computes the sequence length, and then loops over percentages. Taking absolute value is important because the paper discusses selecting components by ABSOLUTE IMPORTANCE, not only positive contributions. That is also consistent with the EAP-IG literature they build on.

For each percentage:
- it converts percent into a number of MLP neurons and attention heads,
- calls get_top_scoring_components(...),
- concatenates the selected heads and MLP neuron-components into a circuit,
- evaluates that circuit with circuit_faithfulness(...),
- stores the result in a faithfulness matrix.

The diagonal computation reflects the standard setup where the same percentage is used for both heads and MLP neurons. 

Still need to look into the commented-out off-diagonal code.

The saved file path:
faithfulness_{metric}_{l or vl}_node_circuit.pt
matches the paper’s modality-specific circuit analysis.

#### parse_args()

`parse_args()`
This exposes the experimental knobs: model name/path, seed (default = 42), task name, --language_only, AP-IG integration steps.

`--language_only` is the code counterpart of the paper’s “textual variant vs visual variant” setup.

`--ap_ig_steps` (Number of steps for EAP-IG) defaults to 5, matching Equation 1, where the paper explicitly says k = 5, following Hanna et al.

#### main()
`main()` is the end-to-end paper pipeline in one place.

It: loads the model, loads and splits the dataset, computes or loads AP-IG node scores, runs faithfulness analysis across circuit sizes.

The lines
```
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
model.set_use_hook_mlp_in(True)
```
tell the TransformerLens fork to expose the extra hooks needed for mechanistic analysis on VLMs. Appendix B.3 says they use their own TransformerLens fork with VLM patching support; these flags are part of that instrumentation.