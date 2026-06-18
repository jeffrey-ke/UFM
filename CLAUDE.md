# CLAUDE.md — UFM (UniFlowMatch)

Guidance for Claude Code when working in this repository.

## What this is

UFM ("Unified Flow & Matching") is a transformer model that directly regresses a dense
pixel-displacement field (optical flow) **plus** a covisibility mask, and applies one model to
both optical-flow and wide-baseline matching. Paper: *"UFM: A Simple Path towards Unified Dense
Correspondence with Flow"* (`ufm.pdf`, arXiv:2506.09278).

## Reference docs (read these first)

- **[docs/paper-code-map.md](docs/paper-code-map.md)** — full **all-branch** paper→code map: the
  complete paper inventory (concepts, contributions, algorithms, losses, training tricks,
  architectures, stages, I/O), an annotated per-branch codebase index, the per-item mapping table
  with `[branch] file:line` citations, and sections on dead code, code-only mechanisms, paper/code
  divergences, and open questions. **Consult it before reasoning about how any paper claim maps to
  this code** — it records what is implemented-and-used vs. dead/partial/missing, on which branch.
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
