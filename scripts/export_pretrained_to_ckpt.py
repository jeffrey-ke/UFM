"""Export released UFM weights into a Lightning-resumable local .ckpt, bypassing the broken
from_pretrained path.

Why: released checkpoints' config.json carry the pre-rename info-sharing arg `max_num_views_for_pe`,
while the pinned uniception requires `max_num_views` (global_attention_transformer.py), so
`UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-*")` crashes during construction
(train_pl.py:503). This script builds the model from the LOCAL Hydra config (current API — the same
build from-scratch training uses, train_pl.py:489), then loads ONLY the released weights into it. A
constructor-arg rename does not change state-dict parameter/buffer names, so the load matches. The
result is saved as {"state_dict": {"model."+k: v}} — exactly the shape train_pl.py's HF branch builds
(train_pl.py:506) — so the `torch.load` resume branch (train_pl.py:511) consumes it unchanged.

A clean (empty missing/unexpected) load is also the plan's Stage-0 guard: it proves the repo really is
the architecture of `--model-config` (e.g. Large), not Giant/Refine silently best-effort-loaded.

Run:  cd /home/jeffk/repo/UFM-train
      env -u PYTHONPATH uv run python scripts/export_pretrained_to_ckpt.py \
        --repo infinity1096/UFM-Base-DINOv2L-init --out runs/ufm-base-l.ckpt
"""

import argparse
import os

import torch
from hydra import compose, initialize_config_dir
from huggingface_hub import hf_hub_download, list_repo_files
from omegaconf import OmegaConf

from uniflowmatch.models import (
    UniFlowMatch,
    UniFlowMatchClassificationRefinement,
    UniFlowMatchConfidence,
)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
MODEL_CLASSES = {
    "UniFlowMatch": UniFlowMatch,
    "UniFlowMatchConfidence": UniFlowMatchConfidence,
    "UniFlowMatchClassificationRefinement": UniFlowMatchClassificationRefinement,
}


def load_released_state_dict(repo: str) -> dict:
    """Download the released weights WITHOUT from_pretrained (which would reconstruct from the stale
    config.json and crash). Robust to the weights filename (.safetensors vs .bin)."""
    files = list_repo_files(repo)
    weight_name = next((f for f in ("model.safetensors", "pytorch_model.bin") if f in files), None)
    if weight_name is None:
        cands = [f for f in files if f.endswith((".safetensors", ".bin"))]
        if not cands:
            raise FileNotFoundError(f"No weights file in {repo}; repo files: {files}")
        weight_name = cands[0]
    path = hf_hub_download(repo, weight_name)
    print(f"downloaded {repo}/{weight_name}")
    if weight_name.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="infinity1096/UFM-Base-DINOv2L-init",
                    help="HF repo of released UFM weights to export")
    ap.add_argument("--out", default="runs/ufm-base-l.ckpt", help="output .ckpt path")
    ap.add_argument("--model-config", default="uniflowmatch_covisibility",
                    help="configs/model/<name> to build locally (must match the released architecture)")
    args = ap.parse_args()

    # Build the model from the LOCAL Hydra config (current uniception API). dataset/loss overrides
    # only satisfy default.yaml's `???` mandatory fields; they don't affect the model.
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        cfg = compose(
            config_name="default",
            overrides=[f"model={args.model_config}", "dataset=optflow", "loss=robust_epe_covisibility"],
        )
    model_args = OmegaConf.to_container(cfg.model.model_args, resolve=True)
    model = MODEL_CLASSES[cfg.model.model_class](**model_args)
    print(f"built {cfg.model.model_class} from configs/model/{args.model_config}")

    # Load ONLY the released weights (param names are stable across the max_num_views rename).
    # strict=False so we can inspect the diff; a shape mismatch still raises (wrong architecture).
    sd = load_released_state_dict(args.repo)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Some params are registered under multiple names via shared modules (e.g. the DPT reassembly
    # convs are `scratch.layerN_rn` == `scratch.layer_rn.N` == `input_process.N.1`). The checkpoint
    # stores one alias; loading it populates the real tensor, but the OTHER alias names show up as
    # "missing". Those are not real gaps — detect them by shared storage (data_ptr) with a loaded param.
    msd = model.state_dict()
    missing_set = set(missing)
    loaded_ptrs = {msd[k].data_ptr() for k in msd if k not in missing_set}
    truly_missing = [k for k in missing if msd[k].data_ptr() not in loaded_ptrs]
    aliased = [k for k in missing if k not in truly_missing]

    if unexpected or truly_missing:
        print("\n!! NON-EMPTY DIFF — released weights do not cleanly fit this architecture:")
        print(f"  truly missing ({len(truly_missing)}): {truly_missing[:10]}{' ...' if len(truly_missing) > 10 else ''}")
        print(f"  unexpected    ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
        raise SystemExit(
            "Refusing to export a partial model. If this repo is a different architecture "
            "(e.g. UFM-Base-DINOv2L-init is dim-1024), pass a matching --repo/--model-config."
        )
    if aliased:
        print(f"note: {len(aliased)} key(s) reported missing are shared-module aliases of loaded "
              f"tensors (harmless), e.g. {aliased[:2]}")
    print("load OK — every released tensor consumed; all model params populated (complete match)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {"state_dict": {"model." + k: v for k, v in model.state_dict().items()}}
    torch.save(ckpt, args.out)
    print(f"saved {args.out}  ({len(ckpt['state_dict'])} tensors, keys prefixed 'model.')")
    print(f"resume with:  resume_model={args.out}")


if __name__ == "__main__":
    main()
