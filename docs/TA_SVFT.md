# Task-Adaptive SVFT (TA-SVFT)

This implements the supplied successor specification, not a measured accuracy claim.
The original language training/evaluation workflows remain baseline workflows. The new
method is integrated into `vision_experiments/finetuning_setup.py`; its controller and
Trainer can also wrap ordinary unquantized Linear modules in other models.

## Installation and first experiment

Use a clean environment. Install a matching PyTorch/torchvision pair for the platform,
then `python -m pip install -r requirements-ta-svft.txt`. TA-SVFT and the native SVFT
path need no `torch_sparse`, PEFT, DeepSpeed, vLLM, or online metric download.
Other existing PEFT methods still require their own PEFT dependencies.

From the project root:

```bash
OMP_NUM_THREADS=1 python -m pytest -q
bash vision_experiments/run_ta_svft.sh
```

The second command downloads ViT-B ImageNet-21K weights and CIFAR-100 and uses the
existing 10-training-examples/class and 2-validation-examples/class protocol.
Change `--model_name vit-large` or `--dataset_name food101`, `flowers102`, `resisc45`.
DINOv2's existing model option is also supported. The original train/validation/test
splitting rules and image resize/normalization remain intact; calibration uses only
the selected training split, never validation/test examples.

The launcher is an example configuration, not tuned hyperparameters. Repeat with
`--seed 0`, `--seed 1`, etc., and different output directories for multiple seeds.

## Mathematics and budget

For PyTorch `weight[out, in]`, economy SVD gives `U[out,r]`, `V[in,r]` (the transpose
of returned `Vh`), `r=min(out,in)`. The original weight stays frozen and explicit:

`W' = W0 + U_tilde (diag(a) + S) V_tilde.T`.

All `r` diagonal coefficients start at zero; no spectrum is truncated. Exactly
`B=ta_off_budget` scalar slots form one global Parameter. Per-layer row, column and
slot-index buffers assign those slots to distinct off-diagonal atoms. All slots
are assigned after calibration, even when their learned numerical values are zero.
Thus **support cardinality** and the number of numerically nonzero values differ.

`P_adapter = sum(r_l) + B`; disable the diagonal for the S-only ablation.
The classifier is trained and reported separately. Frozen bases, singular values,
indices, saliency snapshots, and original weights are buffers, never optimizer
parameters. The report separates adapter/head/total trainable parameters. Generic
`total_p` includes original model weights and adaptation scalars, excluding extra
basis storage; `frozen_buffer_bytes` measures floating buffer storage separately.

The forward applies `X V_tilde`, sparse `M`, then `U_tilde.T` without constructing
a dense update matrix. `spectral_update()` and merging explicitly materialize it.
Zero coefficients preserve the original function. Sparse COO coalescing provides
coefficient gradients; inactive candidate coefficients do not exist as Parameters.

## Task information and rectangular matrices

Calibration computes the exact scalar minibatch-mean-loss gradient `G=dL/dW'`
using a temporary, unregistered weight leaf for one layer at a time. It computes
`E_batch[(U_tilde.T @ G @ V_tilde)^2]`, not the square of the mean gradient. Global
Top-B selection covers all eligible matrices. Ties use path then row-major order.
Calibration runs in eval mode and replays Torch/Python/NumPy RNG states, restores
training modes/trainability, and leaves training RNG unchanged. The factory must
return a fresh iterator; calibration batches are equally weighted minibatches.

For tall layers, the mean-gradient residual is `Gbar-U@(U.T@Gbar)`; the top left
singular directions augment U. For wide layers the transposed analogue augments V.
Numerical zero directions are rejected using a tolerance relative to Gbar. New
vectors are reprojected and orthonormalized, and frozen thereafter. At most p
vectors are retained, also limited by ambient complement dimension and residual
rank. The diagonal remains the original r entries; complement edges compete for B.
A second calibration pass scores the augmented dictionary. Complements are created
once and serialized, not recomputed during refinement or loading.

Calibration retains dense statistics for one layer, plus at most B global candidate
records. This limits memory but repeats model forwards/backwards for each target;
it is **not free** and can be slow for all-module ViT-L. Default selection is
one-shot to avoid unmeasured recurring cost. No decomposition cache is shared across
runs; SVD runs once per target on a fresh attachment and is skipped by adapter load.

## Configuration

| Flag | Default | Meaning |
|---|---:|---|
| `--finetuning_method` | `head` (legacy) | Set `ta_svft` explicitly |
| `--ta_families` | `q k v o up down` | HF ViT/DINOv2 projection families across depth |
| `--target_modules` | empty | For TA-SVFT: exact Linear paths, overrides families |
| `--ta_off_budget` | 1024 | One global off-diagonal budget, excludes diagonal/head |
| `--ta_diagonal` | true | Full economy-spectrum diagonal safety path |
| `--ta_complement_rank` | 4 | Maximum frozen task complement directions; 0 disables |
| `--ta_calibration_batches` | 8 | Training minibatches per gradient-estimation pass |
| `--ta_selection` | gradient | `gradient` or seeded matched-budget `random` ablation |
| `--ta_update_interval` | 0 | Optimizer steps between refinement; 0 one-shot |
| `--ta_freeze_step` | 1000 | Refine only at steps strictly below this value |
| `--ta_replace_fraction` | 0.1 | Global fraction replaced, rounded down |
| `--ta_basis_dtype` | float32 | float32 or float64 decomposition/basis/coefficient math |
| `--ta_detailed_support` | false | Export edges, coefficients, saliency and evolution |
| `--ta_adapter_checkpoint` | empty | Warm-start saved adapter and head, new optimizer |
| `--resume_from_checkpoint` | empty | Full Trainer checkpoint including optimizer/scheduler/RNG |
| `--other_learning_rate` | 1e-4 (legacy) | Adapter LR; example launcher uses 0.01 |
| `--clf_learning_rate` | 1e-3 | Separate head LR |

For the dynamic variant, e.g. append `--ta_update_interval 100 --ta_freeze_step 800`.
Use `--ta_families q v`, `--ta_families up down`, or all families for allocation
ablations. Random selection still uses task-derived complements unless
`--ta_complement_rank 0` is also supplied. Equal-per-layer allocation and spectral
block selection are not implemented; the latter was a future research idea, not a
specified core mechanism. Fixed band/random baseline comparisons remain available.

## Refinement and optimizer semantics

The concrete prune criterion is absolute active coefficient magnitude (unit-norm
atoms). Every update removes floor(B*fraction) globally weakest edges and regrows
highest-saliency **previously inactive** edges. If insufficient inactive atoms
exist, the strongest of the removed edges are retained. B remains exact. Surviving
coordinates retain coefficient values, slot identity and optimizer moments even
when other slots move between layers. Reused slots start at zero and their Adam
first/second/AMSGrad moments or SGD momentum are cleared. Adam's scalar step counter
remains global to the bank, as in dense masked dynamic sparse training; regrown
slots do not get an independent Adam bias-correction clock.

The Trainer callback runs after complete optimizer steps, respecting gradient
accumulation. Stock Trainer constructs the scheduler from actual optimizer steps.
Supported refinement optimizers are unsharded Adam, AdamW and SGD. The supplied
vision CLI uses separate head/adapter AdamW groups. Current Trainer integration
requires one process/device; DDP, FSDP, DeepSpeed, model parallelism, compilation,
and gradient checkpointing are rejected explicitly. Ordinary eager FP32 is the
validated training mode; CUDA/mixed-precision performance is unmeasured. Keep the
spectral buffers in configured precision; do not blanket-cast an attached model
with `.half()`/`.bfloat16()`.

## Saving, loading, merging

A completed vision run writes `ta_adapter.pt`, `ta_support.json`, a standard merged
Hugging Face model and image processor in `merged/`, and experiment result JSON.
Periodic Trainer checkpoints contain full adapted state plus `ta_config.json` and
safe tensor/primitive RNG state in `ta_rng.pt`. Resume requires the same TA-SVFT
configuration and target paths; support and complement shapes are restored directly.
Stock Hugging Face `from_pretrained` does not inject custom layers automatically.
Use the controller to load adapters or use the merged model for plain HF inference.

```python
from svft.ta_svft import TASVFT, TASVFTConfig, vit_targets

# model must already contain the intended pretrained weights and classifier shape.
controller = TASVFT(model, vit_targets(model), TASVFTConfig())
# Train through TASVFTTrainer, or initialize with controller.select_support(factory, loss_fn).
controller.save_adapter("adapter.pt")
controller.export_report("support.json", include_edges=True)

# fresh_model must use the SAME original pretrained weights and head architecture.
loaded = TASVFT.load_adapter(fresh_model, "adapter.pt")
deploy_model = loaded.merged_copy()  # leaves the trainable adapted model intact
# loaded.merge() performs a terminal in-place replacement for deployment.
```

Adapter files include frozen U/V/complements (SVD signs/degenerate bases must be
preserved), spectrum, selected support, global coefficients and the complete head,
but exclude original target weights/biases and the rest of the backbone. They can
therefore still be sizable; "adapter-only" does not mean coefficient-only. Loading
needs no task data or SVD, and uses `weights_only=True`. Full-object pickling is not
supported; use state dictionaries/controller APIs. After merging, no support bank,
spectral bases, or selection machinery remain in the deployed model.

## Research outputs

`ta_support.json` records per-layer diagonal/off-diagonal/complement counts,
allocation by projection family and depth, selected index-distance and singular-gap
statistics, refinement count, parameter counts and SVD/support wall-clock timings.
Detailed mode adds `[row, column, global_slot]`, coefficients and last selection
saliency; dynamic updates write `support-STEP.json`. In random ablations saliency
contains random scores. Spectrum gaps are only defined for original spectral
indices, so complement edges are counted separately. Trainer reports training time.
Timings are host wall-clock measurements, not GPU-profiler/FLOP measurements.

## Baseline compatibility and known baseline behavior

Select `--finetuning_method svft --target_modules query value --svft_rank 4` for
Q/V banded SVFT, or `--svft_rank 0` for diagonal/plain. `--svft_pattern` selects
banded, random or top_k. Existing experiment launchers remain present.
The native COO implementation reproduces the original update equation and duplicate
random-coordinate summation without requiring the old torch_sparse extension.
Legacy initialization is deliberately retained: random coefficients with a sigmoid
gate initially 0.5, **not** a zero update. Its random support can contain duplicates;
legacy top_k retains the repository's original heuristic. These are baseline facts,
not TA-SVFT behavior. Vision's undefined post-training `reset_from_svft` call is
replaced by the existing fusion utility. Result-path/TrainOutput serialization and
scheduler step-count bugs are also corrected.
