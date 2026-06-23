# AGENTS.md — UFM-train

Guidance for AI agents working in this repository.

## What this repo is

**UFM-train** is the **`train` branch of a fork of [UniFlowMatch/UFM](https://github.com/UniFlowMatch/UFM)** (`jeffrey-ke/UFM`), packaged as a standalone [uv](https://docs.astral.sh/uv/) project. It trains **UFM (Unified Flow & Matching)** — a DINOv2-based transformer that regresses **dense pixel displacement (optical flow) plus a covisibility mask** between two views.

In refseg-workspace, that correspondence is the **grasp-point identification** mechanism for stage 4: warp a known grasp pixel from the reference image to its location in the observation.

Paper: *"UFM: A Simple Path towards Unified Dense Correspondence with Flow"* (`ufm.pdf`, arXiv:2506.09278).

This checkout is the **training superset**: PyTorch Lightning + Hydra (`scripts/train.py` → `uniflowmatch/training/train_pl.py`), losses, dataset loaders, and the Hydra config tree under `configs/`. Model bodies (DINOv2 encoder, global-attention transformer, DPT heads) live in the git-pinned **`uniception`** package, not in this tree.

## Significance in refseg-workspace

`UFM-train` is the **10th submodule** in [`refseg-workspace`](../README.md) — a meta-repo for **goal-conditioned suction-grasp pose prediction from a single reference image** on box- and can-shaped objects.

### Purpose: grasp-point identification (stage 4)

The full perception stack has four stages. Stages 1–3 are **reference-prompted instance segmentation** (find *where* the target object is). **UFM powers stage 4: locate a specific point on that object in the observation image.**

Given a **reference image** of the target (grasp-anchored canonical view) and an **observation image** (live camera), UFM predicts a **dense correspondence field** (flow + covisibility). A known pixel in the reference — the **grasp point** — is warped through that field to its location in the observation. That observation pixel (plus depth and intrinsics from the segmented point cloud) is the **grasp-point identification** step that feeds 6-DoF suction-cup pose estimation.

```
reference image (grasp-anchored canonical view; grasp pixel known)
  ├─→ [stages 1–3] propose → verify → segment → instance mask on observation
  ▼
observation RGB-D
  │ 4. grasp-point ID  [UFM-train]  ref↔obs dense flow → warp grasp pixel → (x,y) on observation
  │    (then: depth + surface normal at that pixel → 6-DoF suction pose in camera frame)
  ▼
6-DoF grasp pose
```

**This repo trains and fine-tunes that correspondence model** — not the segmentation proposers (those are `reference_matching` / GIM) and not the full 6-DoF head (still in design elsewhere). At inference, the exported UFM checkpoint is the warp engine: *reference grasp pixel in → corresponding observation pixel out*.

Training data reflects this intent: isaac optflow renders use **grasp-anchored reference views** (`OptFlowObject` / `class_to_reference` from per-object grasp frames), and `OptFlowUFMAdapter` builds **1-to-1** ref↔obs pairs so the model learns to track one specific instance's surface correspondence in clutter — exactly the warp needed to carry a grasp point from reference to observation.

### Shared infrastructure with the rest of the workspace

| Shared piece | Role |
|---|---|
| **`isaac_datagen` optflow renders** | Synthetic ref↔obs RGB-D + camera poses for dense-matcher fine-tuning |
| **`shelf-optflow` dataset** | Render umbrella pulled via `dspull shelf-optflow`; also used by verifier/segmenter fine-tuning |
| **`vision_core`** | Sibling editable dep (`../vision_core`); dataset contract types (`OptFlowSample`, `ObsMask`, …) and transforms |
| **Artifact registry (`art`)** | `.artifacts.yaml` maps datasets → `jeffke613/refseg-datasets` |
| **Nested `ObsMask` on optflow renders** | Same render dirs feed UFM training **and** seg stages 2/3 (`add_proposals`, `add_inlier_data`) |

Optflow datagen is **dual-purpose**: one Isaac render pass produces UFM grasp-correspondence training data **and** feeds seg stages 2/3 on the same dirs.

### Workspace layout constraints

- **Own venv** — `uv sync` here; **do not unify** with `segmentation/` or `isaac_datagen/` (deliberately incompatible torch/Isaac stacks).
- **Launch cwd** must be this repo root — Hydra paths and `${hydra:runtime.cwd}` resolve from here.
- **`env -u PYTHONPATH`** on every launch — strips Isaac-Sim torch from `PYTHONPATH` if present.
- Submodule pin in `refseg-workspace` should track tested SHAs for reproducible runs.

## Data path and adapter

Training data for refseg fine-tuning:

```
UFM-train/datasets/shelf-optflow/     # dspull shelf-optflow (or symlink to a data disk)
  render{idx:03d}/                    # self-contained render dirs from isaac_datagen
```

`OptFlowUFMAdapter` (`uniflowmatch/datasets/optflow_isaac.py`) converts isaac's **1-to-many** optflow format into **1-to-1** (reference, single-instance) pairs using `iid_mask` to suppress same-class siblings. UFM receives per-view `img`, `depthmap`, `camera_intrinsics`, `camera_pose` (OpenCV cam2world); **flow + covisibility GT are computed on-GPU at collate** (`flow_postprocessing.py`), not read from disk.

Every depth view must carry `covisible_rendering_parameters` or collate asserts — see `docs/paper-code-map.md` items A2/A3.

## Key configs

| Launch | Purpose |
|---|---|
| `machine=refseg` | Portable paths: `root_data_dir=${hydra:runtime.cwd}/datasets` |
| `dataset=optflow` | Generic optflow training (multi-AR, color jitter) |
| `dataset=optflow_finetune` | Shelf-optflow fine-tune: 85/15 train/val split, fixed ~16:9 AR, no photometric aug |
| `training_scheme=finetune` | Light-touch LRs, short schedule, `symmetrize_inputs=False` |
| `loss=robust_epe_covisibility` | Flow + covisibility supervision (required) |
| `resume_model=infinity1096/UFM-Base` | Public pretrained checkpoint (auto-downloads from HF) |

Fine-tune launch:

```bash
cd UFM-train
uv sync
dspull shelf-optflow
env -u PYTHONPATH uv run python scripts/train.py \
    machine=refseg dataset=optflow_finetune training_scheme=finetune \
    loss=robust_epe_covisibility \
    resume_model=infinity1096/UFM-Base
```

Config files:

- `configs/training_scheme/finetune.yaml`
- `configs/dataset/optflow_finetune.yaml`
- `configs/dataset/optflow_isaac/train/finetune.yaml` — `imgnorm` (no color jitter)
- `configs/dataset/quantity_options/{train,val}/optflow_finetune.yaml` — `[:2490]` / `[2490:]` split

⚠ The **2490 split index is data-coupled** — recompute against `len(OptFlowUFMAdapter(...))` if `shelf-optflow` grows (new render dirs change the pool size).

## Module map (refseg-relevant subset)

| Path | Role |
|---|---|
| `uniflowmatch/datasets/optflow_isaac.py` | `OptFlow2UFM` + `OptFlowUFMAdapter` — isaac optflow → UFM pairs |
| `uniflowmatch/training/train_pl.py` | Lightning loop, symmetrization, flow GT collate |
| `uniflowmatch/models/ufm.py` | `UniFlowMatchConfidence` and variants |
| `uniflowmatch/models/lora.py` | `LoRAUniFlowMatch` — PEFT adapter path (`model=uniflowmatch_lora`) |
| `scripts/train.py` | Hydra CLI entry |
| `scripts/export_pretrained_to_ckpt.py` | Build Lightning `.ckpt` from released HF weights (resume workaround) |
| `configs/` | Hydra config tree |
| `.artifacts.yaml` | `art` registry: datasets only (base model is public HF) |

## Pitfalls agents should know

1. **Resume strict-load footgun** — Wrong checkpoint architecture → `train_pl.py` silently falls back to `strict=False`, leaving the encoder randomly initialized. Verify with `scripts/export_pretrained_to_ckpt.py` / empty missing-unexpected keys before long runs. Correct base for default config: `infinity1096/UFM-Base` (dim 768), not necessarily `UFM-Base-DINOv2L-init`.
2. **Do not modify `vision_core` datastructs** for UFM-only needs — extend in the adapter layer here.
3. **Do not unify venvs** across refseg-workspace submodules.
4. **Covisibility GT is collate-time** — adapter bugs surface in `flow_postprocessing.py`, not in `_get_views`.
5. **Export for downstream use** — Lightning `.ckpt` ≠ HF layout; use `save_pretrained` / export scripts before `from_pretrained` in other repos.

## Where to look next

- **`CLAUDE.md`** — detailed module index, data flow, branch layout, working notes
- **`docs/paper-code-map.md`** — first stop for paper→code mapping and training pipeline questions
- **`configs/dataset/optflow_isaac/README.md`** — optflow adapter knobs and UFM eval DSL
- **`.docs_claude/plans/`** — active/completed plans (fine-tune, LoRA, workspace integration)

When touching model or dataset code that changes how a paper item is implemented, keep `docs/paper-code-map.md` in sync.
