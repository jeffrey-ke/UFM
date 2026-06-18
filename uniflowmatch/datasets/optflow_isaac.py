"""Isaac-render optical-flow dataset → UFM training pairs (1-to-1, instance-masked).

Two layers:

* ``OptFlow2UFM`` — framework-agnostic. A **dataset directory** of ``isaac_datagen`` optflow renders
  (``<dataset>/render{idx:03d}/``, each render dir self-contained with its own idx-0 ``OptFlowMetadata``
  and per-frame nested-ObsMask ``OptFlowSample``s; the cross-repo contract lives in
  ``vision_core.datastructs``) becomes lazy **1-to-1** ``(reference, single-instance)`` pairs. The on-disk
  dataset is 1-to-many (one canonical reference per class warps into every same-class instance); we split
  it to 1-to-1 here using each frame's ``iid_mask`` to isolate one instance.
* ``OptFlowUFMAdapter`` — wraps the core in a ``BaseStereoViewDataset`` and maps each ``MaskedUFM`` into
  UFM's 2-view dict. The base resizes + scales K and computes flow/covisibility on-GPU in
  ``on_after_batch_transfer``; we only supply two **cam2world (OpenCV)** views.

All poses are OpenCV cam2world by construction (isaac_datagen Plans 1-2) and handed to UFM as-is — UFM
inverts the other view itself, so we never invert here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from vision_core.datastructs import ObsMask, OptFlowSample, OptFlowMetadata, count_samples
from vision_core.transforms import StripAlpha

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset


@dataclass
class MaskedUFM:
    """One 1-to-1 unit at NATIVE resolution and native types (the UFM wrapper resizes + scales K and
    converts to UFM's numpy-HWC layout). All poses OpenCV cam2world; ``ref2obs`` derived for RoMa-style
    consumers (UFM uses the two cam2world poses directly).

    The ``obs_*`` color/depth are the **full** observation frame (background + every object intact);
    ``obs_invalid_mask`` flags only the OTHER same-class instances — the confusable identical twins of
    this one. The core does NOT decide how it is applied — each consumer chooses: black / background-fill
    the RGB there so the model can't match the reference to a sibling, and zero the depth there so
    flow/covisibility GT can't land on one. Background + other classes are deliberately left untouched
    (their depth disagrees with projected reference points, so UFM's covisibility rejects them on its
    own). The reference is already object-isolated (``ref_depth`` is 0 off-object)."""

    ref_rgb: tv_tensors.Image     # (3, Ha, Wa) RGB
    ref_depth: torch.Tensor       # (Ha, Wa) metric z, 0 off-object
    ref_K: np.ndarray             # (3, 3)
    ref_c2w: np.ndarray           # (4, 4) cam2world, OpenCV
    obs_rgb: tv_tensors.Image       # (3, H, W) RGB (alpha resolved by the injected transform), FULL frame
    obs_depth: torch.Tensor         # (H, W) metric z, FULL frame (raw; consumer suppresses obs_invalid_mask)
    obs_invalid_mask: tv_tensors.Mask  # (H, W) bool, True = an OTHER same-class instance — suppress these
    obs_K: np.ndarray               # (3, 3)
    obs_c2w: np.ndarray             # (4, 4) cam2world, OpenCV
    cls: str
    iid: int
    render: str                   # render-dir name this unit came from (for labeling)

    @property
    def ref2obs(self) -> np.ndarray:
        return np.linalg.inv(self.obs_c2w) @ self.ref_c2w


class OptFlow2UFM(Dataset):
    """A dataset dir of optflow renders → lazy 1-to-1 (reference, single-instance) pairs.

    ``dataset_dir`` holds many ``render{idx:03d}/`` subdirs (the isaac_datagen dataset pattern); each render
    dir is self-contained with its OWN idx-0 ``OptFlowMetadata``, so we deserialize per-dir and concatenate
    ``(render_dir, frame, iid)`` keys across all of them.

    ``rgba_to_rgb`` is an injected ``torchvision.transforms.v2`` transform turning the RGBA observation
    (``obsmask.obs``) into RGB: ``StripAlpha()`` by default (eval), or a background compositor (e.g.
    ``vision_core.transforms.RandomBackgroundComposite``) for train-time domain randomization.
    """

    def __init__(self, dataset_dir, rgba_to_rgb: Callable = None, min_px: int = 2048):
        self.root = Path(dataset_dir)
        self.rgba_to_rgb = rgba_to_rgb or StripAlpha()                           # injected v2.Transform
        self.render_dirs = sorted(
            d for d in self.root.glob("render*") if (d / "class_to_l2w").is_dir()
        )
        if not self.render_dirs:
            raise FileNotFoundError(
                f"No optflow render dirs (render*/ with class_to_l2w/) under {self.root}"
            )
        self._md = {}                                                            # render_dir -> OptFlowMetadata
        self._cn = {}                                                            # render_dir -> {iid: (cls, l2w row)}
        self.index = []                                                          # (render_dir, frame, iid) keys
        for rd in self.render_dirs:
            md = OptFlowMetadata.deserialize(0, rd)                              # per-dir metadata
            self._md[rd] = md
            name_to_iid = {nm: i for i, nm in md.obsmaskmeta.iid_to_name.items()}   # invert once, per dir
            self._cn[rd] = {
                name_to_iid[nm]: (c, n)
                for c, names in md.class_to_name.items()
                for n, nm in enumerate(names)
                if nm in name_to_iid
            }
            for f in range(count_samples(rd, field="obs")):
                iid_mask = ObsMask.deserialize_field(f, rd, "iid_mask").numpy()  # cheapest field; flat layout
                ids, cnts = np.unique(iid_mask, return_counts=True)             # present instances, exactly
                self.index += [
                    (rd, f, int(i)) for i, c in zip(ids, cnts)
                    if int(i) in self._cn[rd] and c >= min_px
                ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k) -> MaskedUFM:
        rd, f, iid = self.index[k]
        md = self._md[rd]
        c, n = self._cn[rd][iid]
        s = OptFlowSample.deserialize(f, rd)

        iid_mask = s.obsmask.iid_mask.numpy()
        siblings = [i for i, (cc, _) in self._cn[rd].items() if cc == c and i != iid]  # other same-class iids
        invalid = np.isin(iid_mask, siblings)                                    # (H, W) bool: suppress these
        return MaskedUFM(
            ref_rgb=md.class_to_reference[c],                                    # already (3, Ha, Wa) Image
            ref_depth=md.class_to_reference_depth[c].float(),
            ref_K=md.class_to_ref_intrinsics[c].numpy().astype(np.float32),
            ref_c2w=(md.class_to_l2w[c][n].numpy() @ md.class_to_ref_pose[c].numpy()).astype(np.float32),
            obs_rgb=self.rgba_to_rgb(s.obsmask.obs),                            # RGBA → RGB, injected; FULL frame
            obs_depth=torch.as_tensor(np.asarray(s.observation_depth, np.float32)),  # raw, full frame
            obs_invalid_mask=tv_tensors.Mask(torch.from_numpy(invalid)),        # explicit; the consumer applies it
            obs_K=np.asarray(md.obs_intrinsics, np.float32),
            obs_c2w=np.asarray(s.cam2world, np.float32),
            cls=c, iid=iid, render=rd.name,
        )


class OptFlowUFMAdapter(BaseStereoViewDataset):
    """Wraps ``OptFlow2UFM``; maps each ``MaskedUFM`` into UFM's 2 view dicts.

    UFM's ``_get_views`` wants ``img`` as numpy HWC uint8 and ``depthmap`` as numpy HW
    (``_crop_resize_if_necessary`` → ``ImageList``/``PIL.Image.fromarray`` → the base normalizes
    HWC→CHW). So we convert the native CHW ``tv_tensors.Image`` at this boundary. ``camera_pose`` is
    cam2world; UFM inverts the other view in ``flow_postprocessing``.
    """

    def __init__(self, *args, dataset_dir, rgba_to_rgb=None, min_px=2048,
                 covis_params=(0.1, 0.1, 0.005), suitable_for_refinement=False, **kwargs):
        self.inner = OptFlow2UFM(dataset_dir, rgba_to_rgb=rgba_to_rgb, min_px=min_px)
        # [absolute τ_d, temperature, relative τ_r] for the depth-reprojection covisibility GT
        # (flow_postprocessing's static pathway). REQUIRED on every depth view: a view missing this key
        # matches no PATHWAY in collate_fn_with_delayed_flow_postprocessing and the collate asserts.
        # [0.1, 0.1, 0.005] is the value every stock depth dataset uses (scannetpp/blendedmvs/…).
        self.covis_params = np.asarray(
            covis_params if covis_params is not None else (0.1, 0.1, 0.005), np.float32)
        # Also a REQUIRED static-pathway key (set per-dataset, not by the base). Gates refinement-stage
        # data selection; irrelevant to base flow/covisibility training. Default False (the common value).
        self.suitable_for_refinement = suitable_for_refinement
        super().__init__(*args, **kwargs)
        self.is_metric_scale = True
        self.is_synthetic = True

    def __len__(self):
        return len(self.inner)

    def _get_views(self, idx, resolution, rng):
        p = self.inner[idx]                           # MaskedUFM (native resolution + types)
        views = []
        for rgb, depth, invalid, K, c2w, tag in (     # CHW tv_tensors.Image → HWC uint8 numpy for UFM
            (p.ref_rgb, p.ref_depth, None,              p.ref_K, p.ref_c2w, f"{p.cls}#ref"),
            (p.obs_rgb, p.obs_depth, p.obs_invalid_mask, p.obs_K, p.obs_c2w, f"{p.cls}#{p.iid}"),
        ):
            rgb = rgb.permute(1, 2, 0).numpy().astype(np.uint8)
            depth = depth.numpy().astype(np.float32)
            if invalid is not None:                   # suppress OTHER same-class instances (this consumer's
                m = invalid.numpy().astype(bool)      # policy, not the core's): black the RGB so the model
                rgb[m] = 0                            # can't match the reference to a sibling, and zero the
                depth[m] = 0.0                        # depth so flow/covisibility GT can't land on one.
            rgb, depth, K = self._crop_resize_if_necessary(rgb, depth, K, resolution, rng)
            view = dict(
                img=rgb, depthmap=depth, camera_pose=c2w, camera_intrinsics=K,
                dataset="OptFlowIsaac", label=f"{self.inner.root.name}/{p.render}", instance=tag,
                is_widebaseline=True, is_synthetic=True,
                suitable_for_refinement=self.suitable_for_refinement,
            )
            view["covisible_rendering_parameters"] = self.covis_params   # (3,) f32; collate stacks → (B,3)
            views.append(view)
        return views
