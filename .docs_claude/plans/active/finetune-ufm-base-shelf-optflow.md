**Status: ACTIVE — approved, not yet executed (2026-06-18).** Fine-tune pretrained UFM-Base (Large) on the
isaac_datagen `shelf-optflow` renders via the verified optflow→UFM adapter. Follows the optflow adapter
smoke-test (see `isaac_datagen/.docs_claude/plans/completed/optflow-ufm-training-smoketest.md` +
`optflow-ufm-smoketest-fixes.md`). Stages: 0 resume-verify → 1 configs → 1b disk viz → 2 launch → 3 export.

---

# Plan — Fine-tune UFM-Base (Large) on shelf-optflow

## Context

The optflow→UFM adapter is verified end-to-end (smoke test passed). Next we **fine-tune the pretrained
UFM-Base on the synthetic `shelf-optflow` data** to specialize it for the reference-prompted-segmentation
domain. "Fine-tune" = resume real pretrained weights and train at a low LR — not the from-scratch smoke run.

Decisions (confirmed with user):
- **Architecture: Large.** The default `model=uniflowmatch_covisibility` is DINOv2-**L** (dim 1024, 12-layer/
  dim-768 transformer). We resume a **Large** UFM-Base checkpoint. ⚠ The only *cached* checkpoint is
  `UFM-Base-DINOv2G-init` (**Giant**, dim 1536) — resuming that into the Large config fails `strict=True` and
  `train_pl.py:513-517` **silently falls back to `strict=False`**, leaving the encoder randomly initialized.
  So Stage 0 verifies a Large checkpoint that strict-loads before we launch.
- **Light-touch LRs** (≈10× below base, mirroring the paper's UFM-980 fine-tune): encoder `1e-6`, heads
  `1e-5`, **uncertainty head frozen** (`lr 0`), ~6 passes over the data, cosine with ~10% warmup,
  `supervision_range=occlusion` (`loss=robust_epe_covisibility`).
- **15% held-out val** via disjoint slices of the single 2930-unit pool (verified count).

Env: torch `2.11.0+cu128`, GPU 1 free (GPU 0 has another process). Batch 2 (+ symmetrization → 8 images @
560) fit 24 GB in the smoke run; the Large model (~447M) + AdamW fits comfortably.

## Stage 0 — resume-verify guard (do this FIRST; it's the footgun)

Before the long run, confirm the chosen checkpoint strict-loads into the Large config (no silent
random-init). Heredoc / `scripts/verify_resume.py`:
1. Hydra-compose `default` with `dataset=optflow_finetune training_scheme=finetune loss=robust_epe_covisibility`.
2. Build the configured model: `MODEL_CLASSES[cfg.model.model_class](**cfg.model.model_args)`.
3. `ref = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base-DINOv2L-init")` (self-describes its
   architecture), `sd = {"model."+k: v for k,v in ref.state_dict().items()}`.
4. `missing, unexpected = model.load_state_dict(sd, strict=False)` → **assert both are empty** (and shapes
   match). Print the result.
- If it asserts non-empty (checkpoint is Giant / wrong), try `infinity1096/UFM-Base` next; whichever yields
  an empty diff is the one to resume. (Network: this downloads the Large checkpoint — the Giant is cached but
  the Large is not.)

## Stage 1 — config files (UFM-train; mirror the existing optflow/quantity_options pattern)

**`configs/training_scheme/finetune.yaml`** (LR/epoch/batch knobs):
```yaml
defaults:
  - default                       # keeps lr_scheduler: cosine; with optflow_pairs=2490, iters/epoch=1245
effective_batch_size: 2
num_epochs: 6                     # ≈ 6 passes ≈ 7470 steps (1245/epoch)
warmup_epochs: 0.6                # 0.6 × 1245 ≈ 747 steps ≈ 10% (NOT the smoke's 500k-step warmup bug)
learning_rate:
  encoder: 1.0e-6
  info_sharing: 1.0e-5
  output_head: 1.0e-5
  uncertainty_head: 0             # frozen (lr 0 → requires_grad False), per UFM-980
save_ckpt_every_steps: 2000
save_ckpt_every_epochs: 1
keep_n_epoch_ckpts: 3
validate_every_epochs: 1
```

**`configs/dataset/optflow_finetune.yaml`** (top-level; like `optflow.yaml` but finetune quantity_options):
```yaml
defaults:
  - default
  - override quantity_options/train: optflow_finetune
  - override quantity_options/val: optflow_finetune
resolution_train: ${dataset.resolution_options.560_many_ar}
resolution_val: ${dataset.resolution_options.560_1_33_ar}
```

**`configs/dataset/quantity_options/train/optflow_finetune.yaml`** (train = first 85%, units 0..2489):
```yaml
defaults: [default]
optflow_pairs: 2490               # ≈ one pass over the train slice per epoch
data_subsample_ratio: 1
train_dataset_str: "
  + ${dataset.quantity_options.train.optflow_pairs} @ ${dataset.optflow_isaac.train.dataset_str}[:2490][::${dataset.quantity_options.train.data_subsample_ratio}]"
```

**`configs/dataset/quantity_options/val/optflow_finetune.yaml`** (val = last 15%, units 2490..2929):
```yaml
defaults: [default]
test_dataset_str: "${dataset.optflow_isaac.val.dataset_str}[2490:]"
```
(Train and val adapters build the same deterministic index from the same dir, so `[:2490]`/`[2490:]` are
disjoint by construction. `2490 = floor(0.85 × 2930)`.)

## Stage 1b — pred-vs-GT visualization to disk (so we can SEE it learning, not just loss)

UFM already has pred-vs-GT visualizers (`uniflowmatch/utils/viz.py`): `visualizer=flow_covisible` renders
**predicted vs GT flow** (HSV), **predicted vs GT covisibility masks**, and a **per-pixel EPE error map**.
But `visualizer_step` (`train_pl.py:138`) emits **only to wandb** and early-returns when `disable_wandb`.
Mirror `segmentation`'s disk-dump pattern (`segmentation/.../train.py:391`, `plt.imsave` into `runs/<id>/viz/`)
with a small patch so visuals land on disk without wandb:

```python
# train_pl.py visualizer_step, replacing the `if disable_wandb: return` + wandb loop:
save_dir = None
if self.config["disable_wandb"]:
    save_dir = os.path.join(self.trainer.default_root_dir, "viz", append_name)
    os.makedirs(save_dir, exist_ok=True)
for batch_idx in range(num_samples):
    for name, visualizer in self.visualizers.items():
        image = visualizer.visualize(batch_idx, model_result, batch)   # HWC uint8
        if save_dir is not None:                                       # disk fallback (no wandb)
            import matplotlib.pyplot as plt
            plt.imsave(os.path.join(save_dir, f"{name}_{self.global_step:06d}_{batch_idx}.png"), image)
        else:
            wandb.log({f"{name}_{append_name}": wandb.Image(image)}, commit=True)
```
`flow_covisible.yaml` fires train viz every 200 steps (1 sample) and val every batch (all samples) — cap val
spam by overriding `visualizer.validate.num_samples=2` at launch. (Behavior-preserving when wandb is on.)

## Stage 2 — launch (single GPU 1, background, stable run dir)

```bash
cd /home/jeffk/repo/UFM-train
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True env -u PYTHONPATH \
  nohup uv run python scripts/train.py \
  dataset=optflow_finetune training_scheme=finetune loss=robust_epe_covisibility \
  visualizer=flow_covisible \
  resume_model=infinity1096/UFM-Base-DINOv2L-init \
  dataset.num_workers=4 disable_wandb=true \
  visualizer.validate.num_samples=2 \
  hydra.run.dir=/home/jeffk/repo/UFM-train/runs/ufm-shelf-ft \
  > /home/jeffk/repo/UFM-train/runs/ufm-shelf-ft/train.log 2>&1 &
```
- `model=uniflowmatch_covisibility` is the default (Large) — no override needed.
- **Output dir = `runs/ufm-shelf-ft/`** (pinned, not a timestamp): `config.yaml`, `checkpoints_epoch/`,
  `checkpoints_step/`, `viz/` (pred-vs-GT PNGs), `train.log`.
- **wandb:** off → train loss on stdout + pred-vs-GT PNGs on disk. (Val loss scalar isn't surfaced with
  logger=None; the viz PNGs still show val pred-vs-GT. To get the val *curve*, `wandb login` + drop
  `disable_wandb=true` — then viz also routes to wandb instead of disk.)

## Hyperparameter summary

| knob | value | vs base |
|---|---|---|
| resume_model | `UFM-Base-DINOv2L-init` (Large, strict-verified) | — |
| encoder LR | 1e-6 | base 5e-6 (5× lower) |
| heads LR | 1e-5 | base 1e-4 (10× lower) |
| uncertainty head | frozen (lr 0) | base 1e-4 |
| effective_batch_size | 2 (GPU 1) | base 48 |
| num_epochs / optflow_pairs | 6 / 2490 → ~7.5k steps, ~6 passes | base 48 / 100k |
| warmup_epochs | 0.6 (~10%) | base 10 |
| scheduler / precision / clip | cosine / bf16-mixed / norm@1.0 | same |
| supervision | occlusion (robust_epe_covisibility) | same |

## Stage 3 — export the fine-tuned model for downstream pipelines

Checkpoints land as **Lightning `.ckpt`** (`runs/ufm-shelf-ft/checkpoints_epoch/last.ckpt`; `state_dict` keys
prefixed `model.`). UFM's models are HF `PyTorchModelHubMixin` (`ufm.py:120`) so the prescribed consumption
API is `UniFlowMatchConfidence.from_pretrained(dir)` — which needs the `save_pretrained` layout, NOT a raw
`.ckpt`. One-time export (`scripts/export_finetuned.py`):
```python
import torch; from omegaconf import OmegaConf
from uniflowmatch.models import UniFlowMatchConfidence
args  = OmegaConf.load("runs/ufm-shelf-ft/config.yaml")
model = UniFlowMatchConfidence(**OmegaConf.to_container(args.model.model_args, resolve=True))
ckpt  = torch.load("runs/ufm-shelf-ft/checkpoints_epoch/last.ckpt", map_location="cpu")
sd    = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
model.load_state_dict(sd, strict=True)                       # strict → confirms a clean, complete export
model.save_pretrained("runs/ufm-shelf-ft/hf")                # config.json + model.safetensors
```
Downstream (any repo/pipeline, identical to a released model):
```python
model = UniFlowMatchConfidence.from_pretrained("/home/jeffk/repo/UFM-train/runs/ufm-shelf-ft/hf").eval().cuda()
```
(or `model.push_to_hub("<you>/ufm-shelf-ft")` → `from_pretrained("<you>/ufm-shelf-ft")`.)

## Verification

1. **Stage-0 verify passes** — empty missing/unexpected keys (resume is genuinely Large, no silent random-init).
2. **In `runs/ufm-shelf-ft/train.log`:** `"Resumed model from HF repo"` present and **no** `"Failed to load
   state dict strictly, trying best-effort"` (that warning = wrong checkpoint → abort).
3. **Fine-tuning, not from-scratch:** initial train loss starts **far below** the from-scratch smoke (~96) —
   resumed UFM-Base should open in the low tens / single digits and decrease. A ~96 start means resume failed.
4. **Runs to completion:** 6 epochs (~7.5k steps), checkpoints under `runs/ufm-shelf-ft/checkpoints_*`.
5. **Pred-vs-GT visuals:** `runs/ufm-shelf-ft/viz/` PNGs (flow / covisibility / error) show the prediction
   converging toward GT over training — the qualitative signal loss numbers can't give.
6. **Export round-trips:** Stage 3 `save_pretrained` → `from_pretrained` loads back with `strict=True`
   (a complete, reusable model).
7. **(wandb on)** held-out val loss tracked per epoch; if val rises while train falls → overfitting, pick the
   best-epoch checkpoint.

## Caveats
- **Val leakage is mild, not zero:** the slice holds out *later frames* (viewpoints), but the same object
  instances/classes appear in train. It tests viewpoint generalization, not unseen objects. For a stricter
  eval, later hold out by instance/class or render a separate val dir.
- `2490` is tied to the current 2930-unit `shelf-optflow/render000`; recompute if the dataset changes.
- These configs are additive (new files); they don't touch base/optflow training. Single GPU 1 chosen because
  GPU 0 is partly occupied; batch 2 is VRAM-proven from the smoke run (headroom exists to try 3).
