# Phase 3 facial-prior inference refinement

This experiment is isolated from Phase 2. Original `MFLM/cli_demo.py` and `webwarriors/experiments/phase2/run_mflm.py` are not modified.

## Prior input
The runner expects per-image prior PNGs at `phase3/results/facial_priors_v2/<image-stem>.png`.
If a prior is missing, the script logs a warning and falls back to the original mask threshold for that image.

## Outputs
- `phase3/results/baseline_masks/`: original MFLM logits thresholded at 0.
- Set `--output-dir` (or the runner's existing output argument) to `phase3/results/refined_masks/` for refined masks.
- `phase3/logs/`: store run logs here.

## Refinement
`refined_logits = raw_logits + alpha * (2 * prior - 1)`; `alpha` defaults to 0.5 and can be set using `PHASE3_PRIOR_ALPHA`.
This is an exploratory inference-time heuristic, not trained weights or LoRA fine-tuning. Validate alpha on a development set; do not tune it on the final evaluation set. The baseline masks are saved before refinement.
