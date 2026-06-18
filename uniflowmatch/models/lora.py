"""LoRA fine-tuning wrapper for UFM (config-selectable, HuggingFace PEFT).

`LoRAUniFlowMatch` is a drop-in `UniFlowMatchConfidence`: on construction it loads a fully-trained UFM
checkpoint and then injects LoRA adapters into the (now pretrained) DINOv2 encoder and the global
info-sharing transformer via PEFT. The DPT prediction heads stay fully trainable. The frozen base +
low-rank adapters are expressed through the existing `get_parameter_groups` / `learning_rate` machinery
(base groups at lr 0), so the model trains through the unchanged Lightning loop -- it looks, trains, and
checkpoints like a regular UFM.

Only the per-block attention/MLP `nn.Linear`s are adapted (`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`);
the encoder's `patch_embed.proj` (Conv2d) and info_sharing's top-level `proj_embed` (input Linear) are
deliberately left frozen.

Load order is **load-then-inject**: PEFT renames a wrapped layer's weight (`...qkv.weight` ->
`...qkv.base_layer.weight`), so the pretrained weights must be loaded BEFORE injection or they would not
match the post-injection keys.
"""

import os
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn

from uniflowmatch.models.ufm import UniFlowMatchConfidence, modify_state_dict

# re.fullmatch over the full module path -- matches only the per-block attn/mlp Linears, so it skips the
# encoder's patch_embed.proj (Conv2d) and info_sharing's top-level proj_embed (input Linear).
_ATTN_MLP_REGEX = r".*\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))$"

# Same drop-list UniFlowMatch.__init__ applies when loading a training checkpoint (ufm.py:208-210).
_DROP_KEYS = {"feature_matching_proj": None, "encoder.model.mask_token": None}


def _lora_config(sub: Dict[str, Any]) -> LoraConfig:
    """Build a per-submodule LoraConfig from a ``{r, alpha, dropout, rslora, init}`` dict."""
    return LoraConfig(
        r=sub["r"],
        lora_alpha=sub.get("alpha", sub["r"]),
        lora_dropout=sub.get("dropout", 0.0),
        use_rslora=sub.get("rslora", True),
        init_lora_weights=sub.get("init", True),
        target_modules=_ATTN_MLP_REGEX,
    )


def _load_full_ufm_weights(model: nn.Module, path_or_repo: str) -> None:
    """Load a full pretrained UFM state dict into ``model`` (strict), BEFORE any LoRA injection.

    Accepts an HF Hub repo id (e.g. ``infinity1096/UFM-Base``) or a local checkpoint path -- either a
    Lightning ``.ckpt`` (``state_dict`` with a ``model.`` prefix) or a ``{"model": state_dict}`` dict.
    """
    if os.path.isfile(path_or_repo):
        ckpt = torch.load(path_or_repo, map_location="cpu")
        if "state_dict" in ckpt:
            # strip the LightningModule "model." prefix and the torch.compile "._orig_mod" artifact
            state_dict = {k[len("model.") :]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
            state_dict = modify_state_dict(state_dict, _DROP_KEYS)
        else:
            state_dict = ckpt["model"]
        model.load_state_dict(state_dict, strict=True)
    else:
        # HF Hub repo id: rebuild the published model from its stored config, then copy weights across.
        reference = UniFlowMatchConfidence.from_pretrained(path_or_repo)
        model.load_state_dict(reference.state_dict(), strict=True)
        del reference


class LoRAUniFlowMatch(UniFlowMatchConfidence):
    """UFM with LoRA adapters on the encoder + info-sharing transformer; DPT heads stay full fine-tuned.

    Indistinguishable to the training loop from a plain UFM (same ``forward``); only ``get_parameter_groups``
    differs, exposing the frozen base and the LoRA adapters as separate optimizer groups. Must be paired
    with ``training_scheme=lora`` so the base ``encoder``/``info_sharing`` groups are given lr 0.
    """

    def __init__(self, *, lora: Dict[str, Any], pretrained_checkpoint_path: Optional[str] = None, **base_args):
        base_args.pop("pretrained_backbone_checkpoint_path", None)  # we load the FULL checkpoint, not the backbone
        super().__init__(pretrained_backbone_checkpoint_path=None, **base_args)

        # (1) load pretrained weights BEFORE injection (PEFT renames wrapped-layer weight keys).
        if pretrained_checkpoint_path is not None:
            _load_full_ufm_weights(self, pretrained_checkpoint_path)

        # (2) inject LoRA in place; PEFT freezes the base params of each injected submodule.
        inject_adapter_in_model(_lora_config(lora["encoder"]), self.encoder)
        inject_adapter_in_model(_lora_config(lora["info_sharing"]), self.info_sharing)

        # 4 adapted Linears per transformer block (attn.qkv, attn.proj, mlp.fc1, mlp.fc2). Fail fast on a
        # bad regex (0 matches) or an accidentally-adapted stem (would make the count non-divisible by 4).
        n_encoder = sum(isinstance(m, BaseTunerLayer) for m in self.encoder.modules())
        n_info_sharing = sum(isinstance(m, BaseTunerLayer) for m in self.info_sharing.modules())
        assert n_encoder > 0 and n_encoder % 4 == 0, f"unexpected encoder LoRA layer count: {n_encoder}"
        assert (
            n_info_sharing > 0 and n_info_sharing % 4 == 0
        ), f"unexpected info_sharing LoRA layer count: {n_info_sharing}"
        print(f"[LoRAUniFlowMatch] LoRA layers -- encoder: {n_encoder}, info_sharing: {n_info_sharing}")

    def get_parameter_groups(self) -> Dict[str, nn.ParameterList]:
        """Six groups keyed to ``training_scheme.learning_rate``, covering every parameter exactly once.

        Base encoder/info_sharing params go in ``encoder``/``info_sharing`` (set lr 0 to freeze + drop from
        the optimizer); the LoRA adapters go in ``encoder_lora``/``info_sharing_lora``; the DPT heads stay
        whole in ``output_head``/``uncertainty_head``. Insertion order is fixed -- Lightning restores AdamW
        state by group index on resume.
        """

        def lora_of(module: nn.Module) -> nn.ParameterList:
            return nn.ParameterList(p for n, p in module.named_parameters() if "lora_" in n)

        def base_of(module: nn.Module) -> nn.ParameterList:
            return nn.ParameterList(p for n, p in module.named_parameters() if "lora_" not in n)

        groups = {
            "encoder": base_of(self.encoder),
            "info_sharing": base_of(self.info_sharing),
            "encoder_lora": lora_of(self.encoder),
            "info_sharing_lora": lora_of(self.info_sharing),
            "output_head": nn.ParameterList(self.head1.parameters()),
            "uncertainty_head": nn.ParameterList(self.uncertainty_head.parameters()),
        }
        assert all(len(g) > 0 for g in groups.values()), "empty parameter group (check the LoRA target regex)"
        covered = {p for g in groups.values() for p in g}
        assert covered == set(self.parameters()), "parameter groups must cover every model parameter exactly once"
        return groups


def merge_lora_(model: nn.Module) -> nn.Module:
    """Merge every LoRA delta into its base weight in place (zero-overhead inference). Returns ``model``."""
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.merge()
    return model
