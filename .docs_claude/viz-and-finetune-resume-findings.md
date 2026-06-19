# UFM viz + fine-tune resume: findings & changes (2026-06-19)

Session goal: surface training visualizations to wandb, stand up a real fine-tune config, and make
resuming a released checkpoint actually work. Below is what we discovered and what changed. Companion
docs: `plans/active/finetune-ufm-base-shelf-optflow.md` (the fine-tune plan — **its checkpoint choice
is now corrected here, see §3**) and `isaac_datagen/.docs_claude/optflow-ufm-smoketest-fixes.md`.

---

## 1. Visualization → wandb (already existed; extended)

- UFM already ships a config-driven viz system that logs `wandb.Image`s during training:
  visualizer library `uniflowmatch/utils/viz.py` → configs `configs/visualizer/*.yaml` → driver
  `train_pl.py:120` `visualizer_step` (cadence gate + `wandb.log`, rank-0 only).
- Selection key is `field_name`: **`batch/*` = an INPUT, `result/*` = a PREDICTION** (same renderer
  class serves both). Panels are composed with `ConcatVisualizer`; `get_visualizer()` maps YAML
  `class:` → class.
- **Color semantics** (for the standard `flow_covisible` panels):
  - Flow (HSV wheel): **hue = motion direction, saturation = magnitude** (white ≈ no motion), scale 25px.
  - Covis/occlusion mask: **white = covisible, black = occluded/out-of-view** (GT vs predicted).
  - EPE heatmap (coolwarm, per-image normalized): **blue = low error → red = high**; occluded = white.
  - EPE outliers (absolute thresholds): purple ≤0.25px, blue ≤0.5, cyan ≤1, green ≤2, yellow ≤5,
    orange ≤10, red >10px; occluded = white.

### Added: warped-image visualization
No warp visualizer existed (`warp_image_with_flow` was unused). Added **`WarpVisualizer`** with two modes:
- `mode=inverse` (gather): warps **img2 → img1's frame** via the flow (reconstructs **img1**; clean).
- `mode=forward` (scatter, new helper `forward_warp_image`): warps **img1 → img2's frame**
  (reconstructs **img2**; leaves black holes where nothing maps — intrinsic to forward warping).
- New config `configs/visualizer/flow_covisible_warp.yaml` adds a `warp_prediction` panel:
  `img1 ‖ img2 ‖ img2→img1 (pred) ‖ img2→img1 (GT) ‖ img1→img2 (GT forward)`.

### wandb entity
`train_pl.py` doesn't pass `entity` (project is hardcoded `"uniflowmatch"`), so runs default to the
logged-in user's default entity. To send runs to the same entity as the `segmentation` repo
(`jeffke613-carnegie-mellon-university`), launch with `WANDB_ENTITY=jeffke613-carnegie-mellon-university`
(env var; no secret in repo). Auth is ambient via `~/.netrc`.

---

## 2. Fine-tune configs created

Four additive files implement the (previously approved-but-uncreated) light-touch fine-tune:
- `configs/training_scheme/finetune.yaml` — cosine, 6 epochs, warmup 0.6, LRs encoder `1e-6` /
  heads `1e-5` / uncertainty `0` (frozen), `effective_batch_size: 2`.
- `configs/dataset/optflow_finetune.yaml` + `quantity_options/{train,val}/optflow_finetune.yaml` —
  shelf-optflow with an 85/15 split via `[:2490]` / `[2490:]`.

**⚠ The split index `2490` is data-coupled** (`floor(0.85 × 2930)` for render000 only). `shelf-optflow`
now also has render001/render003, so recompute it against the current
`len(OptFlowUFMAdapter(dataset_dir=shelf-optflow))` before a real run, or the split won't be 85/15.

**Configs are necessary but not sufficient** — a successful run also needs launch-time/ambient details
not in any config: `resume_model` (the weights), `machine.root_data_dir` (data lives at
`/home/jeffk/data/datasets`, not the repo path), `CUDA_VISIBLE_DEVICES`/`PYTORCH_CUDA_ALLOC_CONF`,
`WANDB_ENTITY`, `~/.netrc` auth, `dataset.num_workers≥1`, `hydra.run.dir`, and the pulled data.

---

## 3. The fine-tune resume blocker — root cause & fix (Option D)

**Symptom:** `resume_model=infinity1096/UFM-Base-DINOv2L-init` crashed at `train_pl.py:503`
(`from_pretrained`) with `MultiViewGlobalAttentionTransformerIFR.__init__() missing 1 required
positional argument: 'max_num_views'`.

**Layered root cause (discovered in order):**
1. **Arg rename.** The pinned `uniception` (`castacks/UniCeption@1cff080`) renamed the info-sharing PE
   arg `max_num_views_for_pe` → `max_num_views` (now required). `from_pretrained` rebuilds from the
   checkpoint's *saved* `config.json` (old name) → missing required arg. The local Hydra config
   (`configs/model/decoder/base_global.yaml`) uses the **new** name, so **from-scratch builds fine** —
   only released-checkpoint construction breaks. (`**kwargs` in the ctor silently swallows the old name.)
2. **Wrong checkpoint (the real issue).** `UFM-Base-DINOv2L-init` is a *different, larger* architecture
   — info-sharing `dim=1024, num_heads=16, indices=[4,8]`, saved with the **old** arg name. It does not
   fit the repo's default config at all (153 shape mismatches). **`infinity1096/UFM-Base` is the correct
   base**: `dim=768, num_heads=12, indices=[5,8]`, **new** arg name → an exact match for the default
   config, and it loads **100% cleanly**.
3. **Harmless alias keys.** Loading `UFM-Base` reports 16 "missing" keys (`scratch.layerN_rn` /
   `scratch.layer_rn.N`), but those are **shared-module aliases** of `input_process.N.1` (one tensor,
   three registered names via `make_scratch`). The checkpoint stores them under `input_process.*`;
   loading populates the real tensors. All params load.

**Fix — Option D (`scripts/export_pretrained_to_ckpt.py`, new):** build the model from the *local*
Hydra config (current API, drift-proof for arg renames), load only the released **weights**, verify
completeness with an alias-aware check (fail only on truly-unloaded params or shape mismatch / extra
keys), and save a Lightning-format ckpt `{"state_dict": {"model."+k: v}}`. This resumes through the
existing `torch.load` branch (`train_pl.py:511`) with **no core-code change**.

```
env -u PYTHONPATH uv run python scripts/export_pretrained_to_ckpt.py \
  --repo infinity1096/UFM-Base --out runs/ufm-base.ckpt
# resume (NOTE: absolute path — Hydra chdir=True):
resume_model=/home/jeffk/repo/UFM-train/runs/ufm-base.ckpt
```

**Verified:** export produced `runs/ufm-base.ckpt` (629 tensors, complete load); `train_pl.py` resumed
it under `strict=True` (log: `Resuming model from checkpoint: …` with **no** "Failed to load state
dict strictly" fallback). The resume path works.

Scope note: this fixes training **resume**. Released-model **inference** via `from_pretrained` is still
broken for the old-arg-name checkpoints under this pin (would need a uniception fork/alias — "Option A").

---

## 4. Gotchas hit (so the next person doesn't)

- **`persistent_workers needs num_workers > 0`** → launch with `dataset.num_workers≥1` (the loader
  hardcodes `persistent_workers=True`). Same as smoke-fix #5.
- **Hydra `chdir=True`** (cwd → run dir): `resume_model` and any other input path must be **absolute**.
- **`runs/` was not git-ignored** (only `/outputs` and `/datasets` were) → added `/runs` to `.gitignore`
  this commit, so the heavy ckpts/logs/wandb files under `runs/` aren't committed.
- **GPU contention:** verification OOM'd because an external `isaac_datagen` process held ~8–13 GiB on
  *both* 4090s. Environmental, not a code issue — re-run the smoke/fine-tune when a GPU frees up
  (the model needs ~12+ GiB at 560 / batch 2 + symmetrization).

---

## 5. Files changed this session
- **New:** `scripts/export_pretrained_to_ckpt.py`; `configs/visualizer/flow_covisible_warp.yaml`;
  `configs/training_scheme/finetune.yaml`; `configs/dataset/optflow_finetune.yaml`;
  `configs/dataset/quantity_options/{train,val}/optflow_finetune.yaml`.
- **Modified:** `uniflowmatch/utils/viz.py` (`WarpVisualizer` + `forward_warp_image` + factory entry);
  `.gitignore` (`/runs`).
- **Untracked artifacts (not committed):** everything under `runs/` (ckpt, logs, wandb run dirs).

## 6. Status / next
- ✅ Viz (incl. warp) logs to wandb under the chosen entity; fine-tune configs created; resume works.
- ⏳ Run the real fine-tune (`dataset=optflow_finetune training_scheme=finetune
  resume_model=/abs/.../runs/ufm-base.ckpt`) once a GPU frees up; first verify `len(adapter)` and fix
  the `2490` split index.
