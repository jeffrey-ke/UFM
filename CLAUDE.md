---
description: UFM (UniFlowMatch) — unified dense correspondence (optical flow + wide-baseline matching) via direct flow regression + covisibility. This `train` branch is the training superset (Lightning + Hydra), packaged as a uv project.
alwaysApply: true
---

# CLAUDE.md — UFM (UniFlowMatch)

Guidance for Claude Code when working in this repository.

## What this is

UFM ("Unified Flow & Matching") is a transformer model that directly regresses a dense
pixel-displacement field (optical flow) **plus** a covisibility mask, and applies one model to
both optical-flow and wide-baseline matching. Paper: *"UFM: A Simple Path towards Unified Dense
Correspondence with Flow"* (`ufm.pdf`, arXiv:2506.09278).

## Reference docs (read these first)

- **[docs/paper-code-map.md](docs/paper-code-map.md)** — ⭐ **FIRST STOP for any training- or
  data-pipeline work, and before reasoning about how any paper claim maps to this code.** Full
  **all-branch** paper→code map: the complete paper inventory (concepts, contributions, algorithms,
  losses, training tricks, architectures, stages, I/O), an annotated per-branch codebase index, the
  per-item mapping table with `[branch] file:line` citations, and sections on dead code, code-only
  mechanisms, paper/code divergences, and open questions. It records what is implemented-and-used vs.
  dead/partial/missing, on which branch. Especially load-bearing for the **covisibility-from-depth GT**
  (items A2/A3, `datasets/base/flow_postprocessing.py`): that GPU pathway — not `_get_views` — is where
  dataset-adapter bugs surface (e.g. every depth view MUST carry `covisible_rendering_parameters` or the
  `PATHWAY_REQUIREMENTS` collate asserts). When you build/modify a dataset adapter, trace it through the
  map's A2/A3 entries before claiming it's training-ready.
- `README.md` — install, CLI, Python API, model-zoo checkpoints.

## The code spans three branches

The implementation is split across branches; the map tags every citation with the branch it lives on:

- **`main`** — inference + model assembly only (this branch). No loss/optimizer/dataset/benchmark code.
- **`origin/train`** — the **superset**: PyTorch Lightning training (`uniflowmatch/training/train_pl.py`,
  `scripts/train.py`), all losses (`uniflowmatch/loss/`), all 15 dataset loaders +
  covisibility-from-depth computation (`uniflowmatch/datasets/`, esp. `base/flow_postprocessing.py`),
  the Hydra config tree (`configs/`), training bash scripts (`bash_scripts/training/`), and a copy of `benchmarks/`.
- **`origin/benchmark`** — evaluation harness (`benchmarks/ufm_benchmarks/`) + eval datasets (ETH3D, DTU, TA-WB).

Read other branches without disturbing `main`:
```bash
git show train:uniflowmatch/loss/epe.py          # one file
git worktree add ../UFM-train train              # whole branch on disk
git worktree add ../UFM-benchmark benchmark
```
Two gaps are absent from **every** branch: **relative-pose evaluation** (paper Table 3) and the
**TA-WB geometric sampler** (only pre-built pairs are downloaded). See map §8.

## `UniCeption/` is an uninitialized submodule (empty) on all branches

The DINOv2 encoder, the global self-attention transformer, the DPT feature processors, and the
prediction-head adaptors are defined in the external `uniception` package. UFM code only *assembles*
them from config. The architecture is pinned by the Hydra configs on `train`
(`configs/model/`: encoder = DINOv2 ViT-L, transformer depth = 12, DPT intermediate layers = [5,8]
i.e. the paper's {6,9,12}, `refinement_range` = 7), but the module bodies and FlashAttention are
off-disk until:
```bash
git submodule update --init        # populate UniCeption
```

**Packaging note (`train` branch):** this branch is a [uv](https://docs.astral.sh/uv/) project, and
`uniception` is declared in `pyproject.toml` as a git dependency pinned to the same commit as the
submodule (`...@1cff080`). So `uv sync` installs `uniception` automatically — `git submodule update
--init` is only needed if you want the source on disk for reference/editing.

## Layout

| Path | Purpose |
|------|---------|
| `uniflowmatch/models/ufm.py` | Core models: `UniFlowMatch`, `UniFlowMatchConfidence` (adds uncertainty/covariance head), `UniFlowMatchClassificationRefinement` (adds local 7×7 refinement). `forward()`, encode/info-sharing/head assembly, refinement attention. |
| `uniflowmatch/models/base.py` | `UniFlowMatchModelsBase`: `predict_correspondences_batched` (the public inference API), input normalization, resolution scaling, flow/covisibility unmapping; output dataclasses. |
| `uniflowmatch/models/unet_encoder.py` | U-Net for refinement fine features (optional, default off). |
| `uniflowmatch/utils/flow_resizing.py` | Inference resolution selection (`AutomaticShapeSelection`) + `unmap_predicted_flow/channels` back to original resolution. |
| `uniflowmatch/utils/geometry.py` | Depth↔3D / projection / pose / quaternion utilities (mostly used by off-branch training code). |
| `uniflowmatch/utils/viz.py` | `warp_image_with_flow`, flow visualization. |
| `uniflowmatch/cli.py` | `ufm` console script (`demo` / `infer` / `test`). |
| `gradio_demo.py`, `example_inference.py` | Demo + example entry points. |

## The one inference path

All entry points converge on:
`from_pretrained` → `predict_correspondences_batched` (`base.py:137`) → `forward` (`ufm.py`) →
encode (shared DINOv2) → `info_sharing` (global attention) → DPT head(s) → optional refinement →
`unmap_predicted_flow` / `unmap_predicted_channels`.

Public API returns `result.flow.flow_output` (B,2,H,W) and `result.covisibility.mask` (B,H,W).

## Working notes

- Checkpoints load from HuggingFace (`infinity1096/UFM-*`). Variants: Base, Refine, Base/Refine-980,
  Base-DINOv2L-init, Base-DINOv2G-init.
- Inputs may be uint8 or float32, BCHW or BHWC; `predict_correspondences_batched` normalizes them.
- Known bug: `ufm demo` (`cli.py:62`) calls `initialize_model(use_refinement=...)` but the function
  takes `required_model_str` — see `docs/paper-code-map.md` §7/§8.
- When you touch model code, keep `docs/paper-code-map.md` in sync if the change alters how a paper
  item is implemented (status, `file:line`, or a divergence).
- Lint/format: `black` + `isort` (line length 120), config in `pyproject.toml`; `pre-commit` hooks available.

## Quick start

```bash
uv sync                                  # create .venv (inference + training stack); add --extra dev for lint/test
uv run ufm test                          # verify install (imports + CUDA visibility)
uv run python example_inference.py --source a.jpg --target b.jpg   # inference example
uv run python scripts/train.py ...       # Hydra-configured training (see bash_scripts/training/ for full commands)
```

`uniception` installs from a git pin (no submodule init needed). `vision-core` is wired as an editable
sibling dep (`../vision_core`) but is not yet imported anywhere — available for future use.

## Module index

Quick map of the `train` superset (the inference-assembly subset is also in the **Layout** table above).
For the authoritative paper→code mapping with `file:line` citations, see `docs/paper-code-map.md`.

| Module | Role | Key exports |
|---|---|---|
| **Models** | | |
| `uniflowmatch/models/ufm.py` | Core architecture: shared DINOv2 encoder → global-attention transformer → dual DPT heads (flow + covisibility) → optional 7×7 refinement; `forward`, symmetrization, state-dict remap. | `UniFlowMatch`, `UniFlowMatchConfidence`, `UniFlowMatchClassificationRefinement` |
| `uniflowmatch/models/base.py` | Inference API + I/O dataclasses: normalization, resolution scaling, flow/covis unmapping. | `UniFlowMatchModelsBase`, `UFMFlowFieldOutput`, `UFMMaskFieldOutput`, `UFMClassificationRefinementOutput` |
| `uniflowmatch/models/unet_encoder.py` · `models/utils.py` | Optional refinement U-Net; meshgrid helper. | `UNet`, `get_meshgrid_torch` |
| **Loss** | | |
| `uniflowmatch/loss/base.py` | Abstract supervision interface. | `SupervisionBase` |
| `uniflowmatch/loss/epe.py` | Flow regression: `RobustRegressionLoss` (Charbonnier, paper-exact) + unused `FlowEPELoss`; masks by covisible/FOV/all. | `RobustRegressionLoss`, `FlowEPELoss` |
| `uniflowmatch/loss/cross_entropy.py` | Covisibility/FOV BCE. | `CrossEntropyLoss` |
| `uniflowmatch/loss/refinement_cross_entropy[_efficient].py` | 7×7 refinement CE (softened 4-point target); memory-efficient variant. | `RefinementCrossEntropyLoss`, `RefinementCrossEntropyLossEfficient` |
| **Datasets** | | |
| `uniflowmatch/datasets/base/flow_postprocessing.py` | ★ Covisibility-from-depth-reprojection (static/kubric/identity pathways), supervision mask `V_covis`, delayed pathway collation. | `flow_occlusion_post_processing`, `collate_fn_with_delayed_flow_postprocessing`, `PATHWAY_REQUIREMENTS` |
| `uniflowmatch/datasets/base/base_stereo_view_dataset.py` | Depth-based 2-view base (intrinsics/pose/pts3d + augmentation stack). | `BaseStereoViewDataset` |
| `uniflowmatch/datasets/base/base_optical_flow_dataset.py` | Precomputed-flow base. | `BaseOpticalFlowDataset` |
| `uniflowmatch/datasets/base/{easy_dataset,batched_sampler}.py` | Dataset composition; aspect-ratio batched sampler. | `EasyDataset`, `AspectRatioBatchedSampler` |
| `uniflowmatch/datasets/{blendedmvs,megadepth,scannetpp,habitat,staticthings3d,hypersim}.py` | Depth→flow loaders (Hypersim is code-only, +1 over paper's 12). | per-dataset classes |
| `uniflowmatch/datasets/{flyingthings3d,flyingchairs,monkaa,spring,hd1k,kubric4d,tartanair_assembled}.py` | Precomputed-flow loaders. | per-dataset classes |
| `uniflowmatch/datasets/{dtu,eth3d}.py` | Eval-only wide-baseline loaders. | `DTU`, `ETH3D` |
| `uniflowmatch/datasets/utils/*.py` | Cropping, flow manipulation (incl. covisible-guided crop), distributed-H5 reader, kubric I/O. | `CovisibleGuidedCropManipulation`, `DistributedH5Reader`, … |
| **Training** | | |
| `uniflowmatch/training/train_pl.py` | Lightning loop: `LightningModel` (forward/steps/loss-agg, differential LR, symmetrization), `LightningDataModule`, Hydra entry. | `LightningModel`, `LightningDataModule`, `train_pl_main` |
| **Utils** | | |
| `uniflowmatch/utils/flow_resizing.py` | Resolution selection + output unmapping + covariance rescaling. | `AutomaticShapeSelection`, `unmap_predicted_flow`, `unmap_predicted_channels` |
| `uniflowmatch/utils/geometry.py` | Depth↔3D / projection / pose / quaternion (mostly off-inference-path). | `depthmap_to_pts3d`, `geotrf`, `quaternion_to_rot_matrix`, … |
| `uniflowmatch/utils/{viz,image,misc,parallel,device}.py` | Flow viz + image warp; image I/O; misc/parallel/device helpers. | `warp_image_with_flow`, … |
| **Entry points** | | |
| `uniflowmatch/cli.py` | `ufm` console script: `demo` / `infer` / `test`. (`demo` has a known arg bug — see Working notes.) | `main` |
| `gradio_demo.py` · `example_inference.py` | Gradio UI; standalone inference example. | `create_demo`, `process_images`; `load_image`, `predict_correspondences` |
| `scripts/train.py` · `scripts/benchmark.py` | Hydra CLI wrappers → `train_pl_main` / `run_benchmark`. | `main` |
| `benchmarks/ufm_benchmarks/` | Eval harness (its own `setup.py` package `ufm_benchmarks`): solutions + dense-corr / optical-flow metrics. | `UFMSolutionBase`, `run_benchmark` |
| **Configs** | | |
| `configs/` (Hydra) | `model/` (DINOv2-L encoder, 12-layer global attn, DPT heads, adaptor maps), `dataset/` (ufm_560_all, ufm_980_highres, …), `loss/`, `training_scheme/` (LR groups, schedulers, torch_compile), `machine/`, `visualizer/`. | — |

## Data flow

```
TRAINING                                     INFERENCE
scripts/train.py (Hydra)                     from_pretrained(model_id)  [or `ufm infer/demo`]
  → train_pl_main(cfg)                         → predict_correspondences_batched(src, tgt)   base.py
  → Lightning{Model,DataModule}                  → normalize + AutomaticShapeSelection (resize)
    • __getitem__ → per-example                  → forward():                                  ufm.py
    • collate_fn_with_delayed_flow_                  • encode src/tgt  (shared DINOv2 ViT-L)
      postprocessing → covisibility by              • info_sharing   (12-layer global attn + view PE)
      pathway (static/kubric/identity)              • DPT flow head  → flow (B,2,H,W)
    • optional batch symmetrization                 • DPT covis head → σ(logits) (B,1,H,W)
  → model.forward (encode→info_sharing→            • optional 7×7 refinement (residual)
    DPT flow + covis [+ refinement])             → unmap_predicted_flow / _channels (→ orig res)
  → loss = 1·RobustEPE(covisible) + 10·BCE      result.flow.flow_output (B,2,H,W),
    [+ refinement CE]                            result.covisibility.mask (B,H,W)
  → AdamW (differential LR) + cosine+warmup
```
External, off-disk: `uniception` (encoder / info-sharing / DPT / adaptors, assembled from config) — git-pinned.

## Where to look next

Documentation, plans, style guidance, and investigation notes live in `.docs_claude/`.

- `.docs_claude/plans/active/` -- plans currently in progress
- `.docs_claude/plans/completed/` -- finished plans
- `.docs_claude/style-and-beliefs/` -- code style and design principles
- `docs/paper-code-map.md` -- authoritative paper→code map (all branches), the first stop for "where is X implemented?"

## Plans & workflow

Plans are first-class artifacts in `.docs_claude/plans/`.

- **Small change** (one file, obvious fix): no plan needed.
- **Medium change** (new feature, wire up a subsystem): lightweight plan in `plans/active/`.
- **Complex change** (new architecture, pipeline redesign): full execution plan with goal, approach, staged checklist, and decision log in `plans/active/`.

Move completed plans to `plans/completed/`.

**Before planning any new implementation:**
1. Read `plans/active/` -- don't duplicate in-progress work.
2. Read `plans/completed/` -- learn from past decisions and avoid re-solving solved problems.
3. Read relevant docs in `.docs_claude/` -- context that shaped the current design.

## Core beliefs

Before planning any implementation, read `/reusable-parts` and apply its guidelines to the design.
