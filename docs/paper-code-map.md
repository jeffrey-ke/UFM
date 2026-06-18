# UFM — Paper → Code Map (all branches)

Maps the paper **"UFM: A Simple Path towards Unified Dense Correspondence with Flow"** (`ufm.pdf`, arXiv:2506.09278, CMU) to its implementation across all three released branches.

## How to read this document

The project is split across **three git branches**, and the code that implements the paper is spread across all of them. Citations below are tagged with the branch they live on:

| Tag | Branch | Contents |
|-----|--------|----------|
| `[main]` | `origin/main` | Inference + model assembly only (the published package). |
| `[train]` | `origin/train` | **Superset**: training pipeline (PyTorch Lightning), all loss functions, all 15 dataset loaders + covisibility computation, Hydra configs, training bash scripts — *and* a copy of the `benchmarks/` tree. |
| `[bench]` | `origin/benchmark` | Evaluation subset: the `benchmarks/` harness + the three eval datasets (ETH3D, DTU, TA-WB) and their download/preprocess scripts. |

A citation like `[train] uniflowmatch/loss/epe.py:252` means the file is at that path **on the `train` branch**. To read it: `git show train:uniflowmatch/loss/epe.py`, or check out a worktree (`git worktree add ../UFM-train train`).

> **Two structural facts that still shape the map:**
> 1. **`UniCeption/` is an uninitialized submodule on every branch.** The DINOv2 encoder, the global self-attention transformer, the DPT feature processors, and the prediction-head adaptors are defined in the external [`uniception`](https://github.com/castacks/UniCeption) package. UFM code only *assembles* them from config. Architecture items are confirmed by the Hydra configs (encoder = ViT-L, transformer depth = 12, etc.) but the *module bodies* and FlashAttention live in UniCeption, off-disk.
> 2. **Relative-pose estimation (paper Table 3) is absent from all three branches.** A grep for `findEssentialMat`/`recoverPose`/`findFundamentalMat`/`RANSAC`/`AUC`/pose-error across `train` and `benchmark` returns nothing. The dense-correspondence and optical-flow benchmarks are fully present; the pose-AUC evaluation is not shipped.

---

## 1. Paper summary

- **One model for two tasks.** UFM (UniFlowMatch) directly regresses a dense pixel-displacement field (u, v) plus a covisibility mask, and applies the *same* model to both optical flow (small motion) and wide-baseline matching (large viewpoint change).
- **Simple generic architecture.** Shared DINOv2 ViT-L encoder → fuse two views with a view-index positional encoding → 12 global self-attention layers → two separate DPT heads (flow + covisibility). No cost volumes; flow is regressed at pixel resolution.
- **Visually-grounded supervision.** Correspondence is supervised **only on co-visible pixels**, computed from depth-reprojection consistency, so the model grounds matches in visual evidence rather than extrapolating geometry into occluded regions.
- **Unified data helps both domains.** Trained jointly on 12 datasets spanning optical-flow and wide-baseline matching; ablations show mutual improvement.
- **Optional refinement + new benchmark.** A separate stage uses local 7×7 neighborhood attention around the regressed flow for sub-pixel accuracy (UFM-Refine). The paper also introduces the **TartanAir Wide-Baseline (TA-WB)** benchmark.

---

## 2. Paper inventory

### Concepts (C#)
| ID | Name | Description | Paper loc | Key symbols |
|----|------|-------------|-----------|-------------|
| C1 | Unified Flow & Matching | One model regresses (u,v) flow for both OF and WB matching | §3.1, Eq (1) | φ₁ ∈ ℝ^{2×H×W} |
| C2 | Visually-grounded correspondence | Supervise only on co-visible pixels | §3.2, Abstract | — |
| C3 | Covisibility mask | Binary mask C₁ = is source pixel visible in target | §3.1, Eq (1) | C₁=σ(C₁^logits) |
| C4 | Direct flow regression (vs cost volume) | Pixel-level DPT regression, not coarse-to-fine | §3.1, App F | — |
| C5 | Refinement by classification | Local 7×7 neighborhood attention refines regressed flow | §3.1, Fig 3, App D | 7×7 |

### Contributions (N#)
| ID | Name | Description | Paper loc |
|----|------|-------------|-----------|
| N1 | Unified training benefits both domains | Joint OF+WB training reaches SoTA on both | §1, Table 5 |
| N2 | Generic transformer + add-on refinement | Simple architecture + plug-in refinement | §1, Table 2 |
| N3 | TA-WB benchmark | New TartanAir Wide-baseline benchmark | §1, App C |

### Algorithms (A#)
| ID | Name | Description | Paper loc | Key params |
|----|------|-------------|-----------|------------|
| A1 | Refinement attention | residual = softmax(Fₛᵀ Fₙ + b)·Gₙ over local 7×7 | §3.1, Fig 3 | 7×7, bias b |
| A2 | Covisibility from depth reprojection | covisible if e < τ_d + τ_r·‖pₛ−O₂‖ (static/scene-flow/rigid variants) | App A, Eq S.1–S.5 | τ_d, τ_r (Table S.1) |
| A3 | Covisibility supervision mask | V_covis = (V₁∩¬F₁) ∪ (F₁∩V_other) | App A.3, Eq S.6 | — |
| A4 | Geometric sampler (TA-WB) | Voxelize, pick source cam+voxel, candidate cams by angle, covis filter | App C, Fig S.1 | 0°–120° bins |
| A5 | Kubric4D sampling weights | Bias toward large viewpoint + time difference | App B, Eq S.7 | w(α), 3600 pairs/scene |

### Losses & objectives (L#)
| ID | Name | Description | Paper loc | Key params |
|----|------|-------------|-----------|------------|
| L1 | L_EPE (robust regression) | Robust Charbonnier on flow, covisible pixels only | Eq (2), Eq (S.9), App H | α=0.5, c=0.24 |
| L2 | L_BCE (covisibility) | Binary cross-entropy on covisibility mask | Eq (3) | — |
| L3 | Total objective | L = L_EPE + 10·L_BCE | §3.2 | covis weight 10 |
| L4 | Refinement cross-entropy | CE with softened bilinear target over 7×7 | App D, Fig S.3 | w₁..w₄ bilinear |

### Training tricks (T#)
| ID | Name | Paper loc | Paper value |
|----|------|-----------|-------------|
| T1 | Differential LR | §3.4 | heads 1·10⁻⁴, encoder 5·10⁻⁶ |
| T2 | AdamW + cosine + warmup | §3.4 | wd 0.05, β=(0.9,0.95), 10% warmup |
| T3 | Batch symmetrization | §3.4 | eff. batch 96 (½ unique) |
| T4 | Covisibility loss ×10 | §3.2 | 10 |
| T5 | Covisible-only supervision | §3.2, App A.3 | — |
| T6 | High-res bootstrapping (UFM_980) | §3.4 | 10× lower LR, 15 epochs, all-pixel |
| T7 | Do **not** freeze encoder | §5, App E | — |
| T8 | Flash-Attention | §3.1 | — |
| T9 | Separate refinement training | App D | LR 1·10⁻⁴, 30 epochs, frozen base |
| T10 | Dataset vetting | §3.3 | drop ARKitScenes |

### Architectures (M#)
| ID | Name | Paper loc | Key params |
|----|------|-----------|------------|
| M1 | DINOv2 ViT-L encoder (shared) | §3.1, Fig 2 | F_E ∈ ℝ^{1024×H/14×W/14} |
| M2 | View-index positional encoding | §3.1 | — |
| M3 | Global self-attention transformer | §3.1, Fig 2 | 12 layers |
| M4 | DPT flow head | §3.1 | layers {6,9,12}+encoder |
| M5 | Separate DPT covisibility head | §3.1 | — |
| M6 | U-Net + fine-feature MLP (refinement) | §3.1, Fig 3 | — |

### Pipeline stages (S#)
| ID | Name | Paper loc |
|----|------|-----------|
| S1 | GT covisibility + correspondence | App A |
| S2 | Pair sampling | §3.3, App B/C |
| S3 | Base training UFM_560 (48 ep, 8×H100, ~4 d) | §3.4 |
| S4 | High-res bootstrapping UFM_980 (15 ep) | §3.4 |
| S5 | Refinement training (30 ep) | App D |
| S6 | Inference | README, §3.1 |

### Environment & I/O (E#)
| ID | Name | Paper loc |
|----|------|-----------|
| E1 | 12 training datasets | Table 1 |
| E2 | Eval benchmarks (ETH3D, DTU, TA-WB, Scannet-1500, Sintel, KITTI, Ego-Exo4D, WxBS) | §4 |
| E3 | Output: flow φ₁ + covisibility C₁ | §3.1 |
| E4 | Input: RGB, longest side 560/980, aspect 3:1–1:1 | §3.4 |
| E5 | TA-WB dataset (TartanAirV2 envs) | App C |

---

## 3. Codebase index (by branch)

### `[main]` — inference + assembly
```
uniflowmatch/
├── models/ufm.py            UniFlowMatch / UniFlowMatchConfidence / UniFlowMatchClassificationRefinement
├── models/base.py           predict_correspondences_batched, normalization, resize, flow/covis unmapping
├── models/unet_encoder.py   UNet for refinement fine features
├── utils/flow_resizing.py   AutomaticShapeSelection + unmap_predicted_flow/channels
├── utils/geometry.py        depth↔3D/projection/pose/quaternion utils
├── cli.py / ../gradio_demo.py / ../example_inference.py   entry points
```

### `[train]` — training (superset)
```
scripts/train.py                         Hydra entry point
uniflowmatch/training/train_pl.py        LightningModel + LightningDataModule + train_pl_main
uniflowmatch/loss/
├── base.py                              SupervisionBase
├── epe.py                               FlowEPELoss, RobustRegressionLoss (Charbonnier)
├── cross_entropy.py                     CrossEntropyLoss (covisibility BCE)
├── refinement_cross_entropy.py          RefinementCrossEntropyLoss (softened 4-point target)
└── refinement_cross_entropy_efficient.py
uniflowmatch/datasets/
├── base/base_stereo_view_dataset.py     depth-based datasets (intrinsics/pose/pts3d, augmentations)
├── base/base_optical_flow_dataset.py    precomputed-flow datasets
├── base/flow_postprocessing.py          ★ covisibility-from-depth-reprojection + V_covis
├── base/batched_sampler.py              DUSt3R-style aspect-ratio batched sampler
├── base/easy_dataset.py                 dataset composition (concat/resample/mul)
├── utils/{cropping,flow_manipulation,...}.py  crops, augmentations, covisible-guided crop
├── {blendedmvs,megadepth,scannetpp,habitat,staticthings3d,hypersim}.py   depth→flow datasets
├── {tartanair_assembled,flyingthings3d,flyingchairs,spring,monkaa,hd1k,kubric4d}.py
configs/   Hydra tree: model/, dataset/, loss/, training_scheme/, machine/, visualizer/
bash_scripts/training/megatraining_{560,980,refinement_560,refinement_980}.sh
benchmarks/   (same tree as [bench], see below)
preprocess_datasets/   per-dataset download + preprocess scripts
```

### `[bench]` — evaluation harness
```
benchmarks/ufm_benchmarks/
├── base.py                                          UFMSolutionBase, iteration/result containers + timer
├── benchmarks/dense_correspondence/
│   ├── widebaseline_datasets.py                     ETH3D/DTU/TA-WB benchmarks + AEPE/outlier metrics
│   └── opticalflow_datasets.py                      Sintel/KITTI benchmarks + EPE/Fl metrics
├── solutions/ufm.py                                 wraps UFM checkpoints as a benchmark solution
└── run_benchmark.py                                 CLI
uniflowmatch/datasets/{eth3d,dtu,tartanair_assembled}.py   eval dataset loaders
preprocess_datasets/download_scripts/download_ta_wb.sh     downloads pre-built TA-WB pairs
```

**Entry points** (ground truth for "used"):
`[main]` `cli.py:13`, `gradio_demo.py:212`, `example_inference.py:93`, `ufm.py:1255` · `[train]` `scripts/train.py` → `train_pl_main` (`training/train_pl.py:438`) · `[bench]` `run_benchmark.py`.

---

## 4. Mapping table

Status: ✅ implemented & used · 🔶 partial/divergent · ⚠ implemented-but-unused · ❌ not found (in any branch).

| ID | Paper item | Paper loc | Implementation (branch:file:line) | Status | Notes |
|----|-----------|-----------|-----------------------------------|--------|-------|
| C1 | Unified Flow & Matching | §3.1 | `[main] ufm.py:120`; trained via `[train] configs/dataset/ufm_560_all.yaml` | ✅ | Architecture + joint-data training both present. |
| C2 | Covisible-only supervision | §3.2 | `[train] epe.py:40-41,71` (`supervision_range="occlusion"` → `non_occluded_mask`) | ✅ | Flow loss masked to covisible pixels. |
| C3 | Covisibility mask | §3.1 | `[main] ufm.py:436-440,665-669`, `base.py:322`; trained by `[train] cross_entropy.py` | ✅ | σ(logits) at inference; BCE-supervised in training. |
| C4 | Direct flow regression | §3.1 | `[main] ufm.py:271-274,425-429` | ✅ | DPT + `FlowAdaptor`; no cost volume. |
| C5 | Refinement by classification | §3.1 | `[main] ufm.py:719,1026`; trained by `[train] refinement_cross_entropy.py` | ✅ | UFM-Refine. |
| N1 | Unified training benefits both | §1 | `[train] configs/dataset/ufm_560_all.yaml` (mixes OF+WB), `train_pl.py` | ✅ | Mechanism (joint 12-dataset training) implemented; the *mutual-improvement* result is experimental. |
| N2 | Generic transformer + refinement | §1 | architecture `[main] ufm.py`; refinement `[main] ufm.py:719` | ✅ | Both present. |
| N3 | TA-WB benchmark | §1, App C | eval: `[bench] widebaseline_datasets.py:188-260`; data: `download_ta_wb.sh`; **sampler: —** | 🔶 | Benchmark *evaluation* + downloadable pairs exist; the *geometric sampler that builds it* (A4) is not shipped. |
| A1 | Refinement attention softmax(FₛᵀFₙ+b)Gₙ | Fig 3 | `[main] compute_refinement_attention ufm.py:1055`, `obtain_neighborhood_features ufm.py:1126`, bias `ufm.py:841` | 🔶 | Faithful; divergences: temperature τ (`ufm.py:1096`, default 4 — not in paper formula) and **bicubic** neighborhood sampling (`ufm.py:1183`). |
| A2 | Covisibility from depth reprojection | App A | `[train] flow_postprocessing.py:544-715` (`flow_occlusion_post_processing`); reproj error `:429-475` (bilinear `grid_sample`) | 🔶 | τ_d=0.1, τ_r=0.005 (per-dataset `covisible_rendering_parameters`). **Divergences:** an extra soft-threshold term `−log(0.5)·temperature` (`:621-627`) beyond Eq S.5, and an optional Adam *iterative* occlusion refinement (`opt_iters`, `:636-688`) not described in the paper. Three pathways (static/kubric/identity) per `PATHWAY_REQUIREMENTS` `:34`. |
| A3 | Covisibility supervision mask V_covis | App A.3 | `[train] flow_postprocessing.py:705-709` | ✅ | Exact `(depth_validity ∧ ¬fov) ∨ (fov ∧ other_depth_validity)` = Eq S.6. FlyingChairs all-zero special case `[train] flyingchairs.py:36-38`. |
| A4 | Geometric sampler (TA-WB) | App C | — (download only: `[bench] preprocess_datasets/download_scripts/download_ta_wb.sh`) | ❌ | The voxelize→candidate-camera→covis-filter→SuperPoint+LightGlue solvability pipeline is **not in code**; only the pre-computed pairs are downloaded. |
| A5 | Kubric4D sampling weights | App B | loader `[train] kubric4d.py:48,93` reads pre-sampled `sampled_pairs.npz` | 🔶 | Loader consumes the *result*; the w(α)/frame-bias sampling algorithm is done offline and **not shipped**. |
| L1 | Robust Charbonnier | Eq (2)/(S.9) | `[train] epe.py:215-329` (formula `:288-291`); params `[train] configs/loss/robust_epe_covisibility.yaml` (α=0.5, c=0.24) | ✅ | Config matches paper exactly. Note: class **default `c=0.03` (`epe.py:217`) is unused** — overridden by config. |
| L2 | L_BCE covisibility | Eq (3) | `[train] cross_entropy.py:100` (`binary_cross_entropy_with_logits`) | ✅ | Optionally masked by `occlusion_supervision_mask` (= V_covis). |
| L3 | Total loss (+10×BCE) | §3.2 | `[train] configs/loss/robust_epe_covisibility.yaml` (mult 1.0 + 10); summed `[train] train_pl.py:63-68` | ✅ | L = 1·L_EPE + 10·L_BCE. |
| L4 | Refinement CE (softened) | App D | `[train] refinement_cross_entropy.py:138-185` (`4_point_average` bilinear target); mult 10 in `robust_epe_covisibility_refinement.yaml` | ✅ | `single` (nearest) strategy `:128-137` is the unused alternative. Matches Fig S.3 weights. |
| T1 | Differential LR (heads 1e-4 / encoder 5e-6) | §3.4 | `[train] configs/training_scheme/default.yaml` (encoder 5e-6, info_sharing/output_head/uncertainty 1e-4); applied `train_pl.py:176-206` (lr=0 ⇒ frozen) | ✅ | Matches paper. |
| T2 | AdamW + cosine + warmup | §3.4 | `[train] train_pl.py:198` (AdamW); `default.yaml` wd=0.05, β=[0.9,0.95], warmup 10%; `lr_scheduler/cosine.yaml` | ✅ | All values match. |
| T3 | Batch symmetrization (eff. batch 96) | §3.4 | `[train] train_pl.py:356-382` (`make_batch_symmetric`, sets `view["symmetrized"]`) | ✅ | Encode-once-interleave; `megatraining_560.sh` `effective_batch_size` × symmetrization ≈ 96 pairs (½ unique). |
| T4 | Covisibility loss ×10 | §3.2 | `[train] configs/loss/robust_epe_covisibility.yaml` (CE multiplier 10) | ✅ | — |
| T5 | Covisible-only supervision range | §3.2 | `[train] epe.py:38-45` (`supervision_range` ∈ {occlusion, fov_mask, all, …}) | ✅ | Base 560 uses `occlusion`; 980 switches to `all` (T6). |
| T6 | High-res bootstrapping UFM_980 | §3.4 | `[train] bash_scripts/training/megatraining_980.sh` + `configs/dataset/ufm_980_highres.yaml` + `configs/loss/robust_epe_all_pixels.yaml` | ✅ | Resumes `UFM-Base`, 15 epochs, LR 10× lower (encoder 5e-7, heads 1e-5), 980 res, all-pixel supervision. (Uncertainty head frozen `lr=0` here — an extra detail.) |
| T7 | Do not freeze encoder | §5, App E | `[train] train_pl.py:191-195` freezes only if `lr==0`; encoder lr=5e-6 in base | ✅ | Encoder trained at low LR (frozen only in refinement-560 stage). |
| T8 | Flash-Attention | §3.1 | `[main] ufm.py:12` comment; actual FA in UniCeption `info_sharing` | 🔶 | Not configurable on-disk; lives in the (empty) submodule. |
| T9 | Separate refinement training | App D | `[train] configs/loss/refinement_only.yaml` (EPE mult 0) + `megatraining_refinement_560.sh` (encoder lr 0, 30 ep) | ✅ | Frozen base + CE-only, 30 epochs — matches paper. (980 refinement instead co-trains EPE+CE with encoder lr 5e-7.) |
| T10 | Dataset vetting (drop ARKitScenes) | §3.3 | curated set in `[train] configs/dataset/ufm_560_all.yaml`; no ARKitScenes loader | 🔶 | The vetting *outcome* (which datasets are included) is in configs; the vetting *process* is not code. |
| M1 | DINOv2 ViT-L encoder (shared) | §3.1 | assembly `[main] ufm.py:196,316-318`; config `[train] configs/model/encoder/dinov2_l.yaml` (embed 1024, patch 14, size large) | ✅ | Config confirms ViT-L; class body in UniCeption. Single encoder over concatenated batch ⇒ shared weights. |
| M2 | View-index positional encoding | §3.1 | `[train] base_global.yaml` (`max_num_views: 2`, `use_rand_idx_pe_for_non_reference_views: False`); plumbing `[main] ufm.py:178` | 🔶 | PE configured into UniCeption's `info_sharing`; body off-disk. |
| M3 | Global self-attention transformer (12 layers) | §3.1 | `[train] configs/model/decoder/base_global.yaml` (`global_attention`, **depth: 12**, dim 768, heads 12); invoked `[main] ufm.py:401` | ✅ | **Confirmed 12 layers.** |
| M4 | DPT flow head (DINOv2 + layers 6/9/12) | §3.1 | `[train] base_global.yaml indices=[5,8]` + `uniflowmatch_covisibility.yaml returned_intermediate_layers:[5,8]`; assembled `[main] ufm.py:405-418` | ✅ | **Resolved:** DPT consumes `[encoder, layer-5, layer-8, final]` = 1-indexed **{6, 9, 12}** + encoder. Exact match to paper. |
| M5 | Separate covisibility head | §3.1 | `[main]` `uncertainty_head` `ufm.py:562-564,665-669`; config `[train] uncertainty_feature_head: dpt` + `uncertainty_adaptor_map: covisibility` (MaskAdaptor) | ✅ | Distinct DPT head; `detach_uncertainty_head: False` in base config (covis grads flow into shared features). |
| M6 | U-Net + fine-feature MLP (refinement) | §3.1, Fig 3 | `[main] UNet unet_encoder.py:36`, fine MLP `MLPFeature ufm.py:1209`, fusion `ufm.py:981-996`; config `[train] uniflowmatch_refinement.yaml` | ✅ | **Confirmed used:** `use_unet_feature: True`, `feature_combine_method: 'modulate'`, `use_unet_batchnorm: True`. |
| S1 | GT covisibility + correspondence | App A | `[train] flow_postprocessing.py` | ✅ | See A2/A3. |
| S2 | Pair sampling | §3.3 | `[train] batched_sampler.py:11-76`; per-dataset pairs (e.g. `scannetpp.py:35` covis>25% pre-filtered) | 🔶 | Within-dataset pairing present; the TA-WB geometric sampler (A4) is not. |
| S3 | Base training UFM_560 | §3.4 | `[train] bash_scripts/training/megatraining_560.sh` (48 ep, `ufm_560_all`, `robust_epe_covisibility`) | ✅ | Matches paper (8×H100/~4-day wall-clock not encoded). |
| S4 | High-res training UFM_980 | §3.4 | `[train] megatraining_980.sh` | ✅ | See T6. |
| S5 | Refinement training | App D | `[train] megatraining_refinement_{560,980}.sh` | ✅ | See T9. |
| S6 | Inference pipeline | §3.1 | `[main] base.py:137`, `flow_resizing.py:618,749` | ✅ | Resolution selection + flow/covis unmapping. |
| E1 | 12 training datasets | Table 1 | `[train] uniflowmatch/datasets/*.py` (+ configs) | ✅ | All 12 present; **+ a 13th, Hypersim (`hypersim.py`), not in the paper's table** (code-only). |
| E2 | Eval benchmarks | §4 | dense corr: `[bench] widebaseline_datasets.py`, `opticalflow_datasets.py`; **pose: —** | 🔶 | ETH3D/DTU/TA-WB/Sintel/KITTI metrics implemented; **relative-pose AUC (Table 3) absent on all branches**; baseline methods not in harness (UFM-only). |
| E3 | Output: flow + covisibility | §3.1 | `[main] UFMFlowFieldOutput base.py:11`, `UFMMaskFieldOutput base.py:24` | ✅ | — |
| E4 | Input: RGB, longest 560/980, aspect 3:1–1:1 | §3.4 | `[main] base.py:86-100` (`AutomaticShapeSelection`); res lists in `[train] ufm_*.yaml` | ✅ | — |
| E5 | TA-WB dataset | App C | loader `[bench] tartanair_assembled.py:17`; download script | 🔶 | Same as N3/A4: data available, sampler not shipped. |

**Status tally (all branches):** ✅ 27 · 🔶 9 · ⚠ 0 · ❌ 2 (A4 geometric sampler, relative-pose evaluation within E2).

> **Correction vs. the earlier `main`-only draft of this file:** items previously marked ❌/⚠ "only because the code wasn't on `main`" (the losses L1–L4, training tricks T1–T7/T9, data S1–S2/A2–A3/A5, datasets E1, benchmarks E2) are now located on `train`/`benchmark` and re-scored. Also, three things flagged as "dead/divergent on `main`" are in fact **used by the released UFM-Refine** (config `uniflowmatch_refinement.yaml`): the U-Net branch, the `feature_combine_method="modulate"` fusion, and `refinement_range=7` (so the 7×7 matches the paper — the `main` class default of 5 is just overridden by config).

---

## 5. Unused / dead implementations

1. **`moge_conv` head type** — `[main] ufm.py:275-276,459`; config exists (`[train] configs/model/feature_head/moge_conv.yaml`) but every shipped model uses `dpt`. No training config selects it.
2. **`UFMClassificationRefinementOutput.log_softmax` on the `[main]` inference path** — `[main] ufm.py:1101,1015-1021` packs it, but inference reads only `flow`/`covisibility`. It exists to feed `[train] RefinementCrossEntropyLoss` (`refinement_cross_entropy.py:75,108`). Used in training, dead at inference.
3. **`FlowEPELoss` (non-robust) and the `single` refinement strategy** — `[train] epe.py:10-212`, `refinement_cross_entropy.py:128-137`. Present as alternatives; shipped configs use `RobustRegressionLoss` and `4_point_average`. (`FlowEPELoss` appears with `multiplier: 0.0` purely as a logging metric in refinement configs.)
4. **`for i in range(1):` single-iteration refinement loop** — `[main] ufm.py:1001`; vestigial multi-step structure.
5. **`from_pretrained_ckpt` + local-file state-dict branches** — `[main] ufm.py:228,207-226`; shipped entry points use HF `from_pretrained`.
6. **Alternative LR schedulers / encoders / adaptors** — `[train] configs/training_scheme/lr_scheduler/{inverse_sqrt,linear_cooldown}.yaml`, `configs/model/encoder/radio_*.yaml` (RADIO), `configs/model/adaptor_map/flow_mol.yaml` (mixture-of-Laplace). Infrastructure for experiments not in the paper; default path uses cosine + DINOv2 + `flow_scale_both`.
7. **Broad geometry/quaternion toolbox** — `[main] geometry.py` (`find_reciprocal_matches:525`, quaternion utils `:545,:584`) — far exceeds what inference uses; supports off-path data/eval code.

---

## 6. Code-only mechanisms (in code, not in the paper)

1. **Uncertainty / flow-covariance / keypoint-confidence head (significant).** `[main] UniFlowMatchConfidence ufm.py:483` can emit `flow_covariance` (`Covariance2DAdaptor`), `keypoint_confidence` (`ConfidenceAdaptor`), with correct covariance rescaling under resize (`base.py:295-319`). Not described in the paper (cf. co-author MAC-VO [47]). The shipped base config wires only the covisibility `MaskAdaptor`, but the covariance/confidence machinery is fully built.
2. **Soft-threshold + iterative occlusion estimation.** `[train] flow_postprocessing.py:621-627` adds a `−log(0.5)·temperature` term to the covisibility threshold (beyond Eq S.5), and `:636-688` runs an Adam optimization (`opt_iters`) to estimate a lower-bound reprojection error. Both are engineering refinements absent from Appendix A.
3. **A 13th dataset: Hypersim.** `[train] hypersim.py` (depth→flow pathway, same `covisible_rendering_parameters`) — not in the paper's Table 1 of 12.
4. **Refinement temperature τ and bicubic sampling.** `[main] ufm.py:1096` (τ, default 4) and `ufm.py:1183` (bicubic `grid_sample`) — see §7.
5. **Training infrastructure not in the paper.** `[train]` `torch.compile` (10+ config modes), gradient clipping (`norm`, 1.0), bf16-mixed precision, `RefinementCrossEntropyLossEfficient`, rich augmentation stack (color jitter, grayscale, blur, low-light darkening, rot90, portrait/landscape, aug_swap, monocular) in `base_stereo_view_dataset.py`. No EMA. `CovisibleGuidedCropManipulation` (`flow_manipulation.py:505`) picks crops by covisibility IoU.
6. **State-dict remapping fossil.** `[main] ufm.py:85,176-182,217-218` drops `feature_matching_proj` and `mask_token` — a remnant of a prior *matching-projection* variant, notable given the paper's regression-over-matching thesis (App F).

---

## 7. Divergences (paper vs. code)

| Topic | Paper | Code | Where |
|-------|-------|------|-------|
| Refinement neighborhood | 7×7 | `refinement_range: 7` ✅ (config overrides the `main` class default of 5) | `[train] uniflowmatch_refinement.yaml:24` vs `[main] ufm.py:756` |
| Refinement attention temperature | none in formula | divides score by τ (default 4) | `[main] ufm.py:1096,749` |
| Neighborhood interpolation | "interpolate features" (Fig 3) | **bicubic** `grid_sample` | `[main] ufm.py:1183` |
| Covisibility threshold | e < τ_d + τ_r·‖pₛ−O₂‖ (Eq S.5) | adds `−log(0.5)·temperature` soft term + optional iterative refinement | `[train] flow_postprocessing.py:621-627,636-688` |
| Charbonnier c | c=0.24 | config 0.24 ✅; **class default 0.03 (unused)** | `[train] epe.py:217` vs `configs/loss/robust_epe_covisibility.yaml` |
| Training set | 12 datasets | 13 loaders (adds Hypersim) | `[train] uniflowmatch/datasets/` |
| UFM_980 uncertainty head | (unspecified) | frozen (`lr=0`) during 980 bootstrapping | `[train] megatraining_980.sh` |
| Refinement loss target | (CE on refinement) | also adds a `RobustRegressionLoss` on `regression_flow_output` in the combined-refinement config | `[train] robust_epe_covisibility_refinement.yaml` |
| `ufm demo` CLI | should launch | **broken**: `launch_demo` calls `initialize_model(use_refinement=...)` but the function takes `required_model_str` → `TypeError` | `[main] cli.py:62` vs `gradio_demo.py:30` |

---

## 8. Open questions / residual gaps

1. **Relative-pose evaluation (Table 3) is in no released branch.** No essential/fundamental-matrix, RANSAC, `recoverPose`, or AUC code exists on `main`, `train`, or `benchmark`. The pose-AUC numbers must come from an unreleased eval script or external tooling. *To confirm:* ask the authors or check release tags.
2. **The TA-WB geometric sampler (A4) and Kubric4D pair sampler (A5) are not shipped** — only their pre-computed outputs (downloaded pairs / `sampled_pairs.npz`). The construction algorithms in App B/C cannot be verified against code.
3. **Baseline-method code (RoMa, MASt3R, SEA-RAFT, FlowFormer, UniMatch, GMFlow)** is not in the benchmark harness; the harness is UFM-only, so the comparative rows in Tables 2–4 were produced outside this repo.
4. **UniCeption internals remain off-disk.** The exact DINOv2 ViT-L wiring, the global-attention block, the DPT processor, FlashAttention usage, and the adaptors are confirmed by config keys but not readable as source until `git submodule update --init`.
5. **`Ego-Exo4D` and `WxBS` qualitative evaluations** (paper §4.3, §5) have no dataset loaders or scripts in any branch — they appear to be one-off qualitative runs.

---

*Generated by the `paper-to-code` skill. `[branch] path:line` citations are durable: read with `git show <branch>:<path>` or a `git worktree`. `train`/`benchmark` worktrees were created at `../UFM-train` and `../UFM-benchmark` during this audit.*
