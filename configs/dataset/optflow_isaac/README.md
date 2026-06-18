# OptFlow → UFM dataset (`dataset=optflow`)

Trains UFM on `isaac_datagen` optical-flow renders via `OptFlowUFMAdapter`
(`uniflowmatch/datasets/optflow_isaac.py`), which turns the 1-to-many optflow dataset into **1-to-1
(reference, single-instance)** correspondence pairs (one same-class instance isolated per sample via its
`iid_mask`).

**Config files:** `configs/dataset/optflow.yaml` (top-level) → `optflow_isaac/{default,train/default,val/default}.yaml`
(the knobs below) + `quantity_options/{train/optflow_100k, val/optflow_full}.yaml` (the mixture string).
`OptFlowUFMAdapter` is exported in `uniflowmatch/datasets/__init__.py` so UFM's `eval()`-based dataset
builder can see it.

## Run

```bash
uv run python scripts/train.py dataset=optflow loss=robust_epe_covisibility ...
```

- `dataset=optflow` — selects this config.
- `loss=...` — UFM marks loss mandatory; `robust_epe_covisibility` supervises flow (robust end-point error)
  + the covisibility/occlusion mask, matching the default `uniflowmatch_covisibility` model.

## How dataset selection works (UFM's eval DSL)

UFM builds the dataset by `eval()`-ing a string (`train_pl.py:332`). After Hydra resolves the `${…}`
interpolations, `train_dataset_str` becomes e.g.:

```
100000 @ OptFlowUFMAdapter(dataset_dir='…/shelf-optflow', resolution=[(560,560),…],
                           transform='colorjitter', data_norm_type='dinov2', min_px=2048)[::1]
```

In UFM's `EasyDataset` mini-language: **`N @ ds`** sets the *per-epoch length to N by oversampling*
(`MulDataset`/`ResizedDataset`); **`[::k]`** subsamples with stride k. So `len == 100000` is the training
**budget/weight**, not the real sample count — the real number of unique 1-to-1 units is whatever the data
yields (e.g. 2930 for the current `shelf-optflow/render000`). Val has no `@ N`, so it is the raw dataset.

## What UFM expects vs. what the adapter builds

UFM is a plain stereo-pair flow learner. Per view, `_get_views` hands it **only**:

| UFM-native (per view dict) | What it is |
|---|---|
| `img` | RGB, numpy **HWC uint8** |
| `depthmap` | metric depth, numpy **HW** |
| `camera_intrinsics` | 3×3 `K` |
| `camera_pose` | 4×4 **cam2world, OpenCV** |

From the **two** views' poses + depth, UFM derives everything else on-GPU in `on_after_batch_transfer`: it
inverts view-B's `camera_pose` to get the relative transform, projects view-A's depth into view-B → dense
**flow + covisibility/occlusion GT**. It is **not** handed a flow map, a precomputed A→B transform, or any
mask.

Everything else is **adapter-side scaffolding** (`OptFlow2UFM`/`OptFlowUFMAdapter`, isaac_datagen-side) that
shapes the renders into that pair and is **gone before UFM sees the batch**:

- **`iid_mask`** (per-pixel instance ids from the render) — used only to (a) enumerate 1-to-1 units,
  (b) apply `min_px`, (c) build `obs_invalid_mask`. Never placed in a view dict.
- **1-to-1 isolation + sibling suppression** — `obs_invalid_mask` (other same-class instances) is **baked
  into `img`/`depthmap`** at `_get_views` time (RGB blacked, depth zeroed there), so UFM just sees a
  modified rgb + depth.
- **RGBA→RGB transform** — resolves the 4-channel observation into the 3-channel `img` UFM wants.

So the knobs below that mention `iid_mask` / masks tune the **adapter**, not the model.

## Knobs (`configs/dataset/optflow_isaac/{train,val}/default.yaml`; override on the CLI)

| Knob | What it controls |
|---|---|
| **`dataset_dir`** | Dataset directory to read. `OptFlow2UFM` globs its `render{idx:03d}/` subdirs (a dataset holds many self-contained render dirs, each with its own idx-0 metadata) and concatenates `(render_dir, frame, iid)` units across them. Default: `…/datasets/shelf-optflow`. |
| **`transform`** | UFM's photometric step *after* resize. Must be one of: `imgnorm` (ImageNet-style normalization only) or `colorjitter` (color jitter + random grayscale + gaussian blur, then normalize — photometric augmentation). Default: `colorjitter` (train) / `imgnorm` (val, deterministic). |
| **`min_px`** | **Adapter-side** (operates on `iid_mask`, which is applied before UFM and never passed to it — see "What UFM expects" above). Minimum `iid_mask` pixel count for an instance to become a training unit. Filters near-invisible instances (heavily occluded / frustum-edge slivers) whose 1-to-1 flow would be near-empty. Higher → fewer/cleaner pairs; lower → more/noisier. Default: `2048`. |
| **`dataset_resolution`** | Resolution bucket key from `resolution_options`. A `*_many_ar` value is a **list** of aspect-ratio buckets `[(560,560),(560,420),…]` (multi-AR training); a single `*_x_y_ar` value is one fixed `(W,H)`. "560" is the base size — use `980_*` for higher-res. Renders (1080×1920) are downscaled to this by `_crop_resize_if_necessary`. Set in `optflow.yaml` via `resolution_train`/`resolution_val` (default `560_many_ar` / `560_1_33_ar`). |
| **`optflow_pairs`** | The `N` in `N @ ds` (`quantity_options/train/optflow_100k.yaml`) — per-epoch sample budget / mixture weight, **not** a real count. With one dataset it just sets epoch length; in a multi-dataset mix, raising it weights optflow higher vs. other datasets. Default: `100_000`. |
| **`data_norm_type`** | Image normalization preset; defaults to `${model.data_norm_type}` (the model decides, e.g. `dinov2`). Leave as-is unless you change the backbone. |
| **RGBA→RGB transform** | `OptFlowUFMAdapter`'s `rgba_to_rgb` (a `torchvision.transforms.v2` transform) converts the 4-ch observation to RGB. Default `StripAlpha` (drops alpha; RGB is already the full frame). Not currently exposed as a config leaf — see below. |

## Background domain-randomization (not wired yet)

To replace the synthetic backdrop with random images (so the model doesn't overfit it), swap the default
`StripAlpha` for `RandomBackgroundComposite(background_dir=…)` (in `vision_core.transforms`). Two gaps to
close first:

1. **Eval-namespace reach.** UFM's eval scope is `from uniflowmatch.datasets import *`;
   `RandomBackgroundComposite` lives in `vision_core.transforms`, so it isn't reachable from the
   `dataset_str` until it's exported into that namespace (or the adapter is built in Python).
2. **Dtype.** It requires/returns float `[0,1]`, but the adapter's `_get_views` converts the image with
   `.astype(uint8)` (expects 0–255). Wrap it:
   `v2.Compose([ToDtype(float32, scale=True), RandomBackgroundComposite(bg_dir), ToDtype(uint8, scale=True)])`.

Until then, `StripAlpha` is the working default.
