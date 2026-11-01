#!/usr/bin/env python3
"""Create an isolated Phase 3 MFLM inference copy with optional facial-prior logit refinement.
Run from any directory; targets the known FakeShield project root.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path('/mnt/DATADRIVE0/Rajnish_B24DS024/ml_project/FakeShield')
PHASE3 = ROOT / 'phase3'
SRC_CLI = ROOT / 'MFLM' / 'cli_demo.py'
SRC_RUNNER = ROOT / 'webwarriors' / 'experiments' / 'phase2' / 'run_mflm.py'

if not ROOT.is_dir():
    sys.exit(f'ERROR: FakeShield project root not found: {ROOT}\nRun this script on the machine that contains the project.')
for p in (SRC_CLI, SRC_RUNNER):
    if not p.is_file():
        sys.exit(f'ERROR: expected source file not found: {p}')

(PHASE3 / 'results' / 'baseline_masks').mkdir(parents=True, exist_ok=True)
(PHASE3 / 'results' / 'refined_masks').mkdir(parents=True, exist_ok=True)
(PHASE3 / 'logs').mkdir(parents=True, exist_ok=True)

# Copy the original inference module; the Phase 2 source remains untouched.
cli_src = SRC_CLI.read_text(encoding='utf-8')
if 'def prepare_mask(' not in cli_src or 'curr_mask = curr_mask > 0' not in cli_src:
    sys.exit('ERROR: cli_demo.py differs from the inspected version; no files were patched.')

# Add imports used by the refinement implementation (only if missing).
if 'from pathlib import Path' not in cli_src:
    cli_src = cli_src.replace('import json\n', 'import json\nfrom pathlib import Path\n')

old = '''            for curr_mask in mask_list:
                color = random.choice(colors_temp)
                if len(colors_temp) > 0:
                    colors_temp.remove(color)
                else:
                    colors_temp = colors
                color_history.append(color)
                curr_mask = curr_mask > 0
                save_img[curr_mask] = (image_np * 0.5 + curr_mask[:, :, None].astype(np.uint8) * np.array(color) * 0.5)[
                    curr_mask]
    seg_mask = np.zeros((curr_mask.shape[0], curr_mask.shape[1], 3), dtype=np.uint8)
    seg_mask[curr_mask] = [255, 255, 255]  # white for True values
    seg_mask[~curr_mask] = [0, 0, 0]  # black for False values
'''
new = '''            for curr_mask in mask_list:
                color = random.choice(colors_temp)
                if len(colors_temp) > 0:
                    colors_temp.remove(color)
                else:
                    colors_temp = colors
                color_history.append(color)

                # Keep the original logit thresholding as a baseline artifact.
                raw_logits = np.asarray(curr_mask, dtype=np.float32)
                baseline_binary = raw_logits > 0
                baseline_dir = Path("phase3/results/baseline_masks")
                baseline_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray((baseline_binary.astype(np.uint8) * 255)).save(
                    baseline_dir / (Path(input_image).stem + ".png")
                )

                # Soft prior refinement: bounded additive bias in logit space.
                # Prior PNG is grayscale [0,255]; mapping to [-1,+1] avoids
                # the extreme penalties caused by log(near-zero prior values).
                prior_dir = Path("phase3/results/facial_priors_v2")
                prior_path = prior_dir / (Path(input_image).stem + ".png")
                alpha = float(os.environ.get("PHASE3_PRIOR_ALPHA", "0.5"))
                if prior_path.is_file() and alpha != 0.0:
                    prior_img = Image.open(prior_path).convert("L").resize(
                        (raw_logits.shape[1], raw_logits.shape[0]), Image.Resampling.BILINEAR
                    )
                    prior = np.asarray(prior_img, dtype=np.float32) / 255.0
                    refined_logits = raw_logits + alpha * (2.0 * prior - 1.0)
                else:
                    refined_logits = raw_logits
                    if not prior_path.is_file():
                        print(f"[Phase3] Prior missing for {Path(input_image).name}; using unrefined logits")

                curr_mask = refined_logits > 0
                save_img[curr_mask] = (image_np * 0.5 + curr_mask[:, :, None].astype(np.uint8) * np.array(color) * 0.5)[
                    curr_mask]
    seg_mask = np.zeros((curr_mask.shape[0], curr_mask.shape[1], 3), dtype=np.uint8)
    seg_mask[curr_mask] = [255, 255, 255]  # white for True values
    seg_mask[~curr_mask] = [0, 0, 0]  # black for False values
'''
if old not in cli_src:
    sys.exit('ERROR: expected prepare_mask block not found; no files were patched.')
cli_src = cli_src.replace(old, new, 1)
(PHASE3 / 'cli_demo.py').write_text(cli_src, encoding='utf-8')

# Isolated runner: import phase3/cli_demo.py before MFLM/cli_demo.py.
runner_src = SRC_RUNNER.read_text(encoding='utf-8')
needle = 'sys.path.insert(0, os.path.abspath("MFLM"))'
if needle not in runner_src:
    sys.exit('ERROR: expected sys.path line not found in run_mflm.py; no runner written.')
runner_src = runner_src.replace(
    needle,
    needle + '\nsys.path.insert(0, os.path.abspath("phase3"))  # prefer isolated Phase 3 cli_demo.py',
    1,
)
(PHASE3 / 'run_mflm_phase3.py').write_text(runner_src, encoding='utf-8')

readme = '''# Phase 3 facial-prior inference refinement

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
'''
(PHASE3 / 'README_refinement.md').write_text(readme, encoding='utf-8')
print('Phase 3 setup complete.')
print(f'Created: {PHASE3 / "cli_demo.py"}')
print(f'Created: {PHASE3 / "run_mflm_phase3.py"}')
print(f'Created: {PHASE3 / "README_refinement.md"}')
print('Original Phase 2 files were not modified.')
