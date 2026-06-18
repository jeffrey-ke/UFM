"""Standalone overfit-one-batch smoke test for the optflow → UFM adapter (no Lightning).

Mirrors uniflowmatch/training/train_pl.py's real training step exactly — model build
(train_pl.py:480-486), losses (train_pl.py:35-37), the on-GPU covisibility GT
(apply_flow_postprocessing_and_merge_batch), and the loss loop (training_step, train_pl.py:58-68) —
but holds ONE fixed batch and loops the optimizer on it. Lightning's overfit_batches can't be used
here: it swaps the custom BatchedRandomSampler for a SequentialSampler (wrong index type + no
set_epoch). This bypasses the dataloader/sampler entirely.

Run:  cd /home/jeffk/repo/UFM-train
      CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH uv run python scripts/smoke_overfit_optflow.py
"""

from dataclasses import replace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from uniflowmatch.datasets.base.flow_postprocessing import (
    apply_flow_postprocessing_and_merge_batch,
    collate_fn_with_delayed_flow_postprocessing,
)
from uniflowmatch.datasets.optflow_isaac import OptFlowUFMAdapter
from uniflowmatch.loss import get_loss
from uniflowmatch.models import (
    UniFlowMatch,
    UniFlowMatchClassificationRefinement,
    UniFlowMatchConfidence,
)

CONFIG_DIR = "/home/jeffk/repo/UFM-train/configs"
DATASET_DIR = "/home/jeffk/repo/isaac_datagen/src/isaac_datagen/datasets/shelf-optflow"
B = 2          # batch size = unique pool (one fixed batch); ViT-L @ 560 is VRAM-heavy
STEPS = 120
MODEL_CLASSES = {
    "UniFlowMatch": UniFlowMatch,
    "UniFlowMatchConfidence": UniFlowMatchConfidence,
    "UniFlowMatchClassificationRefinement": UniFlowMatchClassificationRefinement,
}


def to_cuda(o):
    if torch.is_tensor(o):
        return o.cuda()
    if isinstance(o, dict):
        return {k: to_cuda(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(to_cuda(v) for v in o)
    return o


def main():
    torch.manual_seed(0)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        cfg = compose(config_name="default",
                      overrides=["dataset=optflow", "loss=robust_epe_covisibility"])

    # --- model (mirror train_pl.py:480-486) ---
    model_args = OmegaConf.to_container(cfg.model.model_args, resolve=True)
    model = MODEL_CLASSES[cfg.model.model_class](**model_args).cuda().train()

    # --- losses (mirror train_pl.py:35-37) ---
    supervisions = {
        name: get_loss(v["class"], **OmegaConf.to_container(v["kwargs"], resolve=True))
        for name, v in cfg.loss.train_loss.items()
    }
    print("supervisions:", list(supervisions))

    # --- one fixed batch, single AR, built directly (no DSL / no sampler) ---
    data_norm_type = cfg.model.get("data_norm_type", "dinov2")
    adapter = OptFlowUFMAdapter(dataset_dir=DATASET_DIR, resolution=[(560, 560)],
                                transform="imgnorm", data_norm_type=data_norm_type, min_px=2048)
    print(f"adapter units: {len(adapter)}; taking first {B} as the fixed batch")
    samples = [adapter[(i, 0)] for i in range(B)]
    collated = collate_fn_with_delayed_flow_postprocessing(samples)
    print("pathways:", collated.pathways)
    collated = replace(collated, pathway_batch=to_cuda(collated.pathway_batch))
    batch = apply_flow_postprocessing_and_merge_batch(collated)   # the covisibility GT (A2/A3)
    for d in batch:                                               # mirror on_after_batch_transfer fixups
        d["symmetrized"] = False
        if isinstance(d["data_norm_type"], (list, tuple, set)):
            d["data_norm_type"] = next(iter(set(d["data_norm_type"])))

    # blocker-3 sanity: covisible fraction must be clearly non-zero
    for key in ("non_occluded_mask", "occlusion_supervision_mask", "fov_mask"):
        if key in batch[0]:
            m = batch[0][key].float()
            print(f"  {key}: mean={m.mean().item():.4f}  (shape {tuple(m.shape)})")

    # --- optimizer (mirror configure_optimizers, train_pl.py:176-206; no scheduler) ---
    lrs = cfg.training_scheme.learning_rate
    param_groups = model.get_parameter_groups()
    groups = []
    for name, params in param_groups.items():
        lr = float(lrs[name])
        if lr == 0:
            for p in params:
                p.requires_grad = False
        else:
            groups.append({"params": params, "lr": lr, "name": name})
    opt = torch.optim.AdamW(groups, betas=tuple(cfg.training_scheme.betas),
                            weight_decay=cfg.training_scheme.weight_decay)
    print("lr groups:", {g["name"]: g["lr"] for g in groups})

    # --- overfit loop (mirror training_step, train_pl.py:58-68) ---
    for step in range(STEPS):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            results = model(*batch)
        with torch.autocast("cuda", enabled=False):
            loss_dict = {n: s.compute_loss(batch, results)[1] for n, s in supervisions.items()}
            loss = sum(loss_dict.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == STEPS - 1:
            print(f"step {step:3d}  loss={loss.item():.4f}  " +
                  "  ".join(f"{n}={v.item():.4f}" for n, v in loss_dict.items()))


if __name__ == "__main__":
    main()
