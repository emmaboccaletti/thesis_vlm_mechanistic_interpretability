# MOMENTS Dataset — Inspection Notes from 2026-05-20.
---
## Dataset Location
```
~/thesis_project/data/MOMENTS/
```

---
## Size

| Metric | Count |
|---|---|
| Total clip IDs | 100 |
| Important-moment clips (IM_*.mp4) | 1930 |
| Non-important-moment clips (NIM_*.mp4) | 1942 |
| Important-moment groups per clip | almost always 2 (one clip has 1) |
| v1 JSON files total (IM + NIM) | 3954 |

Class balance is excellent (~50/50 important vs non-important).

---
## Directory Structure

```
MOMENTS/
  {clip_id}/
    important-moments/
      {group_idx}/           # almost always 2 groups per clip (1 and 2)
        IM_1.mp4
        IM_1_v1.json
        IM_1_v1.wav
        IM_1_v2.json
        IM_1_v2.wav
        IM_2.mp4 ... (variable number of clips per group e.g. for 1: IM_8_v2.wav vs 2: IM_4_v2.wav)
    non-important-moments/
      {group_idx}/
        NIM_1.mp4
        NIM_1_v1.json
        ... (variable number of clips per group e.g. for 1: NIM_9_v2.wav vs 2:  NIM_2_v2.wav)
```

Each `IM_*.mp4` / `NIM_*.mp4` is one sample (one brief football event).

<clip_id> (e.g., 0Glu8uEj) = a specific football game

important-moments = moments that appear in the highlight reel

non-important-moments = moments from the full game that do not appear in highlights

1, 2 = two different video sources of the same game
---

## JSON Schema

Both IM and NIM JSONs have identical structure:

```json
{
    "global": "raw ASR transcript run-on no punctuation",
    "local": "Cleaned, punctuated version of the commentary.",
    "similarity": 0.953
}
```

- **`local`** → use this as the text description for prompts (punctuated, readable)
- **`global`** → raw ASR; useful as a corrupted/noisy language variant
- **`similarity`** → cosine similarity between global and local; proxy for transcription quality

### Example — Good quality (IM, similarity=0.953):
```
"global": "Simeone got to it but not well enough back by Clavan Salpedro is underneath it here he is Salpedro holding it up the shot came from Nandez that was high",

"local": "Simeone got to it but not well enough. Backed by Klavan. Salpedro is underneath it. Here he is, Salpedro, holding it up. The shot came from Nandez but was high.",

"similarity": 0.9531517028808594
```

### Example — Poor quality (IM, similarity=0.258):
```
global: "Thank you."
local:  "This is the audio commentary for the video."
```
These are transcription failures — the ASR produced garbage.

### v1 vs v2:
Slightly different time windows or ASR passes of the same event. Descriptions diverge minimally
(different last word or phrase). Both have the same JSON schema.
**Default to `v1`**; `v2` can serve as a second language-corruption variant.

---

## Similarity Score Distribution (v1 JSONs, N=3954)

| Threshold | Count | % |
|---|---|---|
| similarity < 0.3 | 311 | 7.9% |
| similarity < 0.5 | 479 | 12.1% |
| similarity > 0.7 | 2987 | 75.6% |
| similarity > 0.9 | 1711 | 43.3% |

**Filtering recommendation:** exclude samples with `similarity < 0.5`.
This drops ~12% of data but eliminates near-garbage transcriptions.
After filtering: ~1700–1750 IM clips and ~1700–1750 NIM clips remain.

---

## Video Duration

Confirmed from `06FNvY2s/important-moments/1/IM_1.mp4`:
- **FPS:** 30.07
- **Total frames:** 441
- **Duration:** ~14.67s

Extracting 10 uniformly-sampled frames → one frame every ~44 frames ≈ **every ~1.5 seconds**.
This covers the full clip evenly.

---

## Key Design Decisions for Circuit Discovery

### Text description to use
Use `local` field from `_v1.json`. It is clean and punctuated.
Filter: require `similarity >= 0.5`.

### Language corruption
Swap key football terms in the `local` description. Candidate word pairs:
- "goal" ↔ "tackle" / "clearance"
- "shot" ↔ "pass"
- "scores" ↔ "clears"
- "penalty" ↔ "corner"

TODO: The final word list to be determined after annotating goal-specific clips.

Alternatively, use `global` (raw ASR) as the language-corrupted variant for vision-only circuit discovery.

### Vision corruption
Add Gaussian noise to all 10 frames (σ ≈ 0.1 on [0,1] normalized pixel values).

### Frame sampling
Extract 10 frames uniformly from each mp4. Formula: `np.linspace(0, total_frames-1, 10, dtype=int)`.
Keep frame count fixed at 10 across all samples.

### Counterfactual pairing
Each important-moment clip is paired with a non-important-moment clip (different answer).
Follow the same `setup_random_counterfactual_prompts()` pattern as existing tasks.

### Task formulation (binary, starting point)
```
Prompt (visual): [10 frames] + local description + "Is this an important moment in the match? Answer yes or no."
Answer: "yes" for IM, "no" for NIM
```

### Annotation for goal detection (Step 1)
No event-type label exists in the JSON — must be inferred from commentary text or manual review.
Search `local` descriptions for keywords: "goal", "scores", "scored", "net" to shortlist candidates,
then manually verify by watching clips.

---

## Notes

- `non-important-moments` clips use `NIM_*.mp4` naming (not `IM_*.mp4`)
- No event-type labels (goal, penalty, etc.) exist in the dataset — annotation is manual
- Clips per group are **variable**: 06FNvY2s/1 and /2 each have 4, bZhxz4I5/1 has 9. Each IM_*.mp4 is an independent sample.
