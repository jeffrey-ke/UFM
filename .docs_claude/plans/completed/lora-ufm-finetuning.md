# Plan: config-selectable LoRA fine-tuning for UFM (HF PEFT, no vendoring)

## Context

Goal: cheaply **domain-adapt a fully-trained UFM** correspondence model (DINOv2 ViT-L → 12-layer
global self-attention transformer → DPT heads) to the reference-prompted segmentation / suction-grasp
domain, parameter-efficiently and with reduced forgetting (per the LoRA survey §5 "LoRA a correspondence
net" idea). We use HuggingFace **PEFT** rather than a vendored LoRA: UFM is the PEFT-ideal case (uniform
ViT, all `nn.Linear` projections — `qkv`/`proj`/`fc1`/`fc2`, no `nn.MultiheadAttention`), `peft` lands in
UFM-train only (never the shared `vision_core` contract lib), and the user is leery of owning rsLoRA/PiSSA
math by hand.

The bar: the LoRA model must **drop into the existing registry + Lightning loop unchanged** — "looks like
a model, trains like a model, indistinguishable from a regular UFM." It is, because `LightningModel` only
ever touches `model(view1,view2)`, `model.get_parameter_groups()`, `model.parameters()`, `model.compile()`.

Starting point (user-confirmed): a fully-trained UFM checkpoint → LoRA **encoder + info_sharing**
(both are trained weights), **full fine-tune** the small task-specific DPT heads.

### Empirically verified (peft 0.19.1 / transformers 5.12.1 / torch 2.12, via an isolated `uv run` probe)
- `inject_adapter_in_model(cfg, submodule)` **freezes the base itself** (all non-LoRA params in the
  injected submodule, incl. LayerNorms, go `requires_grad=False`); LoRA params named `…lora_A.default` /
  `…lora_B.default`, base weights moved to `…base_layer.weight`.
- **Load-order is mandatory**: loading a normal-key checkpoint into an *already-injected* model gives
  96 missing / 48 unexpected keys (the `attn.qkv.weight → attn.qkv.base_layer.weight` rename). Load
  **then** inject preserves weights bit-for-bit.
- Partitioning param groups by the `"lora_"` substring gives **exact** total coverage of
  `model.parameters()` (satisfies the `train_pl.py:186` assertion).
- Manual merge via `BaseTunerLayer.merge()` round-trips (`forward(adapter)≈forward(merged)`); `unmerge()`
  restores. No `PeftModel` wrapper needed (and `get_peft_model_state_dict(model)` does NOT work on an
  inject-only model — use a manual `"lora_"` filter if adapter-only export is ever wanted).
- A **scoped regex** target cleanly hits attn/mlp linears and excludes the `patch_embed.proj` Conv2d
  (bare-suffix `"proj"` + `exclude_modules` worked but emitted a confusing "no modules excluded" warning).

## How UFM is built today (reuse, don't change)
- Registry instantiation: `train_pl.py:480-486` →
  `MODEL_CLASSES[model_class](**model_args)`, then `LightningModel(raw_model, args)`.
- Hydra composes `configs/model/uniflowmatch_covisibility.yaml` → `model_class` + `model_args`.
- `UniFlowMatchConfidence.__init__` (`ufm.py:483-558`) builds `encoder`/`info_sharing`/`head1`, loads
  backbone via `super().__init__(pretrained_checkpoint_path=pretrained_backbone_checkpoint_path)`, then
  builds `uncertainty_head`; `get_parameter_groups` (`ufm.py:664-682`) returns
  `{encoder, info_sharing, output_head, uncertainty_head}` from `self.<submod>.parameters()`.
- Freeze idiom: a param group with `learning_rate==0` → params `requires_grad=False` + dropped from
  AdamW (`train_pl.py:190-203`). `configure_optimizers` requires every group key to have an LR entry
  (`:179`) and asserts groups cover **all** params (`:186-188`).

## Files to change
1. **`uniflowmatch/models/lora.py`** (new) — `LoRAUniFlowMatch` class + `merge_lora_()` helper + the
   target regex + the full-checkpoint loader. (Keeping it in its own module avoids bloating `ufm.py`;
   it subclasses `UniFlowMatchConfidence`.)
2. **`uniflowmatch/training/train_pl.py`** — add `"LoRAUniFlowMatch": LoRAUniFlowMatch` to `MODEL_CLASSES`
   (`:480-484`). No other training-loop changes.
3. **`configs/model/uniflowmatch_lora.yaml`** (new) — `defaults` same as `uniflowmatch_covisibility.yaml`;
   `model_class: LoRAUniFlowMatch`; `model_args` adds a `lora:` block + a full `pretrained_checkpoint_path`.
4. **`configs/training_scheme/lora.yaml`** (new) — copy of `default.yaml` with `learning_rate` keys:
   `encoder: 0`, `info_sharing: 0`, `encoder_lora: 1.0e-4`, `info_sharing_lora: 1.0e-4`,
   `output_head: 1.0e-4`, `uncertainty_head: 1.0e-4` (differential LoRA LRs allowed per survey).
   Also add **`lora_smoke.yaml`** (same 6 LR keys + `override lr_scheduler: default` (scheduler `null`) +
   `limit_train_batches`/`num_epochs=1`): `training_scheme=smoke` only defines the 4 base LR keys, so it
   `KeyError`s with the LoRA model — the smoke profile must carry the LoRA keys itself.
5. **`pyproject.toml`** — add `peft>=0.13` to UFM-train only (probe-tested against 0.19.1; `transformers`
   already a dep). Confirmed `peft` is not currently installed in any of the four repos.
6. **`bash_scripts/training/`** (optional) — an example `megatraining_lora.sh` launch.

## Key implementation details

**`LoRAUniFlowMatch` (load-then-inject, base frozen, heads full-FT):**
```python
# regex (re.fullmatch on the full param path) hits ONLY block attn/mlp Linears, so it dodges BOTH
# the encoder's patch_embed.proj (Conv2d) AND info_sharing.proj_embed (a top-level input Linear).
_ATTN_MLP_REGEX = r".*\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))$"

def _lora_cfg(sub):   # per-submodule cfg → encoder r can differ from info_sharing r (T7: more capacity on ViT-L)
    return LoraConfig(r=sub["r"], lora_alpha=sub.get("alpha", sub["r"]),
                      lora_dropout=sub.get("dropout", 0.0), use_rslora=sub.get("rslora", True),
                      init_lora_weights=sub.get("init", True), target_modules=_ATTN_MLP_REGEX)

class LoRAUniFlowMatch(UniFlowMatchConfidence):
    def __init__(self, *, lora, pretrained_checkpoint_path=None, **base_args):
        base_args.pop("pretrained_backbone_checkpoint_path", None)        # we load the FULL ckpt, not backbone
        super().__init__(pretrained_backbone_checkpoint_path=None, **base_args)
        if pretrained_checkpoint_path is not None:                        # (1) LOAD pretrained (normal keys)…
            _load_full_ufm_weights(self, pretrained_checkpoint_path)      #     reuse ufm.py modify_state_dict drop-list, strict=True
        inject_adapter_in_model(_lora_cfg(lora["encoder"]), self.encoder)            # (2) …THEN inject; PEFT
        inject_adapter_in_model(_lora_cfg(lora["info_sharing"]), self.info_sharing)  #     freezes base in-place
        # fail-fast targeting check (catches regex/collision regressions, e.g. patch_embed.proj or proj_embed):
        assert sum(isinstance(m, BaseTunerLayer) for m in self.encoder.modules()) == 24*4       # 96
        assert sum(isinstance(m, BaseTunerLayer) for m in self.info_sharing.modules()) == 12*4  # 48

    def get_parameter_groups(self):
        sp = lambda m,k: nn.ParameterList(p for n,p in m.named_parameters() if (k in n)==True)
        lora_of = lambda m: nn.ParameterList(p for n,p in m.named_parameters() if "lora_" in n)
        base_of = lambda m: nn.ParameterList(p for n,p in m.named_parameters() if "lora_" not in n)
        return {                                                          # keys ↔ training_scheme.learning_rate
            "encoder":          base_of(self.encoder),        # lr 0
            "info_sharing":     base_of(self.info_sharing),   # lr 0
            "encoder_lora":     lora_of(self.encoder),        # lr 1e-4
            "info_sharing_lora":lora_of(self.info_sharing),   # lr 1e-4
            "output_head":      nn.ParameterList(self.head1.parameters()),
            "uncertainty_head": nn.ParameterList(self.uncertainty_head.parameters()),
        }
```
- `_load_full_ufm_weights`: reuse the existing remap — HF id → `UniFlowMatchConfidence.from_pretrained(id)`
  (cf. `train_pl.py:499-505`); local `.ckpt` → `torch.load`, strip `model.` prefix, apply
  `modify_state_dict(..., {"feature_matching_proj": None, "encoder.model.mask_token": None})`
  (`ufm.py:204-212`), `load_state_dict(..., strict=True)` **before** injection.
- `merge_lora_(model)`: `for m in model.modules(): isinstance(m, BaseTunerLayer) and m.merge()` →
  zero-overhead inference. (Optional full unwrap to plain `nn.Linear` is a separate export step.)

**`configs/model/uniflowmatch_lora.yaml`** adds to `model_args`:
```yaml
model_class: LoRAUniFlowMatch
model_args:
  # …all existing UniFlowMatchConfidence args…
  pretrained_checkpoint_path: infinity1096/UFM      # or a local .ckpt (HF id or path)
  lora:
    encoder:      { r: 32, alpha: 32, dropout: 0.0, rslora: true, init: true }  # more capacity on ViT-L (T7); bump r→64 if needed
    info_sharing: { r: 16, alpha: 16, dropout: 0.0, rslora: true, init: true }
```
`get_parameter_groups` must return groups in a **fixed insertion order** (Lightning restores AdamW state by
group index when resuming a LoRA run) and assert each `*_lora` group is non-empty (a bad regex → silent
empty adapter set). It already asserts union-coverage to mirror `train_pl.py:186` with a clearer message.

## Verification (end-to-end)
1. **Build/targeting**: instantiate `LoRAUniFlowMatch` from the config; assert `#lora.Linear == 144`
   (encoder 24×4 + info_sharing 12×4), assert `encoder.model.patch_embed.proj` is still `Conv2d`, print
   trainable params (LoRA + both heads only).
2. **Optimizer wiring**: call `configure_optimizers` — no `KeyError` (all 6 group keys have LRs), the
   `:186` coverage assertion passes, `encoder`/`info_sharing` base groups frozen + dropped from AdamW.
3. **Load correctness**: confirm a pretrained encoder weight equals `…base_layer.weight` after
   construction (proves load-then-inject). Confirm **resume_model is NOT used for the base load** (its
   `strict=False` fallback at `train_pl.py:513-518` would silently leave base at random init).
4. **Smoke train** (heed `isaac_datagen/.docs_claude/optflow-ufm-smoketest-fixes.md`):
   `scripts/train.py dataset=optflow loss=robust_epe_covisibility model=uniflowmatch_lora training_scheme=lora_smoke dataset.num_workers=2`.
   - **Scheduler must be OFF** (`lr_scheduler: default` → `scheduler: null`): cosine warmup is sized to
     ~500k steps, so a short run sits at LR≈0 → **flat loss**, a false "LoRA isn't learning" (Fix 6).
   - Cap with `limit_train_batches` + `num_epochs=1`; **never `overfit_batches`** (swaps the custom
     `BatchedRandomSampler` → breaks tuple-index + `set_epoch`, Fix 3). `num_workers>=1` (Fix 5).
   - `CUDA_VISIBLE_DEVICES=<free>`, `effective_batch_size=2`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
     (Fix 7; LoRA saves optimizer/grad memory, **not** activation memory).
   - Confirm `on_after_backward` shows grads only on `lora_*` + heads; `robust_epe` drops quickly.
5. **Merge equivalence**: `forward(adapter) ≈ forward(merge_lora_(copy))` (`allclose`, atol 1e-5).
6. **Checkpoint round-trip**: save a Lightning `.ckpt`, reconstruct `LoRAUniFlowMatch` (re-injects),
   `resume_model` that ckpt (keys now match: `base_layer`+`lora_`) → strict load OK, forward identical.

## Risks / notes
- **`resume_model` trap** (silent-failure, highest risk): the `resume_model` path runs *after* construction
  (i.e. after injection) and does `strict=True` then **falls back to `strict=False`** (`train_pl.py:513-518`).
  Used for the *initial non-LoRA base*, its normal-key weights (`attn.qkv.weight`) become "unexpected" and
  are dropped → encoder trains from ~random. So: load the initial base in the **constructor only**
  (load-then-inject); reserve `resume_model` for resuming a *LoRA run* (whose ckpt already has
  `base_layer`+`lora_` keys → strict match; confirmed the resume path does NOT call `modify_state_dict`,
  so no interference). *Optional hardening:* tighten the fallback to reject missing **non-`lora_`** keys —
  turns the silent base-not-loaded into a loud error.
- **Hydra pairing `KeyError`**: `model` and `training_scheme` compose independently, so a user can pair
  `model=uniflowmatch_lora` with `training_scheme=default`, which lacks the `encoder_lora`/`info_sharing_lora`
  LR keys → `KeyError` at `train_pl.py:179`. Ship `training_scheme/lora.yaml`, document the required pairing,
  and have `get_parameter_groups` (or a small check) raise a clear "use training_scheme=lora" error.
- **torch.compile after inject** (`configure_model` `train_pl.py:284`): shipped default is
  `torch_compile: disabled`, so the default LoRA run won't compile (lowest risk). PEFT `lora.Linear` traces
  fine, but DINOv2's non-reentrant gradient-checkpointing + LoRA is a recompile-prone stack — treat compile
  as opt-in to validate separately. If merging for inference, **merge before compile** (and before publish).
- **bf16-mixed merge tolerance**: `lora_B` inits to 0 (adapter = no-op at step 0). The merge-equivalence
  check must use an fp32 reference or loose tol (`atol=rtol=1e-2`) under `bf16-mixed`.
- **DDP**: existing `find_unused_parameters: True` (`configs/default.yaml` → `train_pl.py:533`) stays. LoRA
  base is *used-but-frozen* (its output feeds the LoRA sum), so it's not "unused"; all trainable params get
  grad. Same idiom UFM already uses to freeze the encoder in the refinement stage.
- **Frozen norms**: PEFT freezes LayerNorms inside encoder/info_sharing. To also train norms/biases
  ("LoRA+norm" trick), add via `modules_to_save` or unfreeze — optional, off by default.
- **HubMixin publish** (only if used): the `lora` dict must round-trip into `config.json` so `from_pretrained`
  re-injects with matching config before `load_state_dict`. Not needed for training.
- **Concurrent UFM work** (`isaac_datagen/.docs_claude/plans/active/ufm-attention-masks.md`): an active plan
  adds opt-in attention masking to the *same* `uniception` `Attention`/`SelfAttentionBlock`. It adds an
  `attn_bias` arg to the attention **forward**; LoRA wraps the `qkv`/`proj` **Linears** — orthogonal, they
  compose. (That plan notes the encoder attention is "frozen-ish" and hard to mask due to torch-hub
  vendoring; LoRA injection is unaffected — it swaps Linear modules, not the forward.)
- **Rejected alternative**: wrapping the whole model in `get_peft_model` reparents the tree to
  `base_model.model.*`, breaking the name-keyed `get_parameter_groups` + LR groups, the `forward`'s
  `self.encoder`/`self.info_sharing` access, and the `model.`-prefix ckpt remap — hence per-submodule
  `inject_adapter_in_model`.

---

## Status — implemented & verified (2026-06-18)

Implemented on branch `train` (UFM-train). As-built deltas vs. the design above: pretrained id defaults to
`infinity1096/UFM-Base` (not `infinity1096/UFM`); `get_parameter_groups` uses local defs (not lambdas); the
count assert uses `n > 0 and n % 4 == 0` (config-robust) and prints the 96/48 totals.

Landed:
- `uniflowmatch/models/lora.py` — `LoRAUniFlowMatch` + `merge_lora_`, `_load_full_ufm_weights`, `_ATTN_MLP_REGEX`.
- `uniflowmatch/models/__init__.py`, `uniflowmatch/training/train_pl.py` — exported + registered in `MODEL_CLASSES`.
- `configs/model/uniflowmatch_lora.yaml`, `configs/training_scheme/{lora,lora_smoke}.yaml`.
- `pyproject.toml` — `peft>=0.13` (synced -> peft 0.19.1 + accelerate; UFM-train only).

Verified through the real Hydra -> registry -> construct path (real DINOv2 ViT-L + UFM-Base):
- strict load-then-inject OK; exactly 96 encoder + 48 info_sharing LoRA layers; `patch_embed.proj` stays Conv2d; `proj_embed` untouched.
- param groups cover all params (`train_pl.py:186` assertion) and keys match the `lora` LR keys (no `:179` KeyError).
- the REAL `configure_optimizers` froze the base groups and built AdamW with only
  `{encoder_lora, info_sharing_lora, output_head, uncertainty_head}` (288 LoRA params + heads trainable; no `base_layer`).
- merge round-trip proven in the isolated peft-0.19.1 probe.

NOT yet run: the end-to-end smoke train (loss-decrease) — blocked on data path + free GPU (below).

## Remaining to run LoRA training on this box (TODO)

Paths NOT set for this box (`jeffk`):
- **optflow dataset_dir is stale.** `configs/dataset/optflow_isaac/{train,val}/default.yaml` default to
  `/home/jeffk/repo/isaac_datagen/src/isaac_datagen/datasets/shelf-optflow`, which does not exist. The actual
  renders on this box are `/home/jeffk/repo/isaac_datagen/datasets/ycb_optflow` (has `render000`); there is no
  `shelf-optflow`.
- **machine config.** Default `machine=yuchen` -> `root_data_dir: /home/inf/match_anything/data` (a different
  box); `machine=default` is `???` (MISSING). For an optflow-only run the dataset_dir is absolute, so
  `root_data_dir` is likely unused — but no machine config matches this box.
- **GPUs (observed 2026-06-18):** GPU 0 at 99% util / ~18.9 GB used (likely an active run); GPU 1 ~11 GB free.

Todos:
- [ ] Point the optflow adapter at the real data — override
  `dataset.optflow_isaac.train.dataset_dir` and `dataset.optflow_isaac.val.dataset_dir` to
  `/home/jeffk/repo/isaac_datagen/datasets/ycb_optflow`, or update the two `optflow_isaac/*/default.yaml` defaults.
- [ ] If anything touches `root_data_dir`, add `configs/machine/<thisbox>.yaml` (or override
  `machine.root_data_dir=...`). Likely not needed for optflow-only.
- [ ] Pick a free GPU via `CUDA_VISIBLE_DEVICES`; ViT-L@560 x2 views x2 symmetrize is heavy — if it OOMs on
  ~11 GB, lower `effective_batch_size` to 1 or drop resolution.
- [ ] Run the smoke train; confirm `robust_epe` drops and grads land only on `lora_*` / heads:
      ```bash
      CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      uv run python scripts/train.py \
        dataset=optflow loss=robust_epe_covisibility \
        model=uniflowmatch_lora training_scheme=lora_smoke \
        dataset.optflow_isaac.train.dataset_dir=/home/jeffk/repo/isaac_datagen/datasets/ycb_optflow \
        dataset.optflow_isaac.val.dataset_dir=/home/jeffk/repo/isaac_datagen/datasets/ycb_optflow \
        dataset.num_workers=2 disable_wandb=True
      ```
- [ ] For a real run: switch `training_scheme=lora_smoke` -> `training_scheme=lora` (cosine + full epochs),
  set `model.pretrained_checkpoint_path` to the UFM checkpoint you want to adapt, and tune
  `effective_batch_size` / `model.lora.encoder.r` to the GPU and task.
