"""
Utility functions for visualization.
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from distutils.util import strtobool
from sklearn.decomposition import PCA
from uniflowmatch.loss import get_loss
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from uniflowmatch.models.base import UFMOutputInterface

def str2bool(v):
    return bool(strtobool(v))


def script_add_rerun_args(parser: ArgumentParser) -> None:
    """
    Add common Rerun script arguments to `parser`.

    Parameters
    ----------
    parser : ArgumentParser
        The parser to add arguments to.

    Returns
    -------
    None
    """
    parser.add_argument("--headless", type=str2bool, nargs="?", const=True, default=True, help="Don't show GUI")
    parser.add_argument(
        "--connect",
        dest="connect",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Connect to an external viewer",
    )
    parser.add_argument(
        "--serve",
        dest="serve",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Serve a web viewer (WARNING: experimental feature)",
    )
    parser.add_argument("--addr", type=str, default="0.0.0.0:2004", help="Connect to this ip:port")
    parser.add_argument("--save", type=str, default=None, help="Save data to a .rrd file at this path")
    parser.add_argument(
        "-o",
        "--stdout",
        dest="stdout",
        action="store_true",
        help="Log data to standard output, to be piped into a Rerun Viewer",
    )


def warp_image_with_flow(source_image, source_mask, target_image, flow) -> np.ndarray:
    """
    Warp the target to source image using the given flow vectors.
    Flow vectors indicate the displacement from source to target.

    Args:
    source_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    target_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    flow: np.ndarray of shape (H, W, 2)
    source_mask: non_occluded mask represented in source image.

    Returns:
    warped_image: target_image warped according to flow into frame of source image
    np.ndarray of shape (H, W, 3), normalized to [0, 1]

    """
    # assert source_image.shape[-1] == 3
    # assert target_image.shape[-1] == 3

    assert flow.shape[-1] == 2

    # Get the shape of the source image
    height, width = source_image.shape[:2]
    target_height, target_width = target_image.shape[:2]

    # Create mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply flow displacements
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    x_new = np.clip(x + flow_x, 0, target_width - 1) + 0.5
    y_new = np.clip(y + flow_y, 0, target_height - 1) + 0.5

    x_new = (x_new / target_image.shape[1]) * 2 - 1
    y_new = (y_new / target_image.shape[0]) * 2 - 1

    warped_image = F.grid_sample(
        torch.from_numpy(target_image).permute(2, 0, 1)[None, ...].float(),
        torch.from_numpy(np.stack([x_new, y_new], axis=-1)).float()[None, ...],
        mode="bilinear",
        align_corners=False,
    )

    warped_image = warped_image[0].permute(1, 2, 0).numpy()

    if source_mask is not None:
        warped_image = warped_image * (source_mask > 0.5)

    return warped_image


def forward_warp_image(image, flow, mask=None) -> np.ndarray:
    """Forward-warp (scatter) the image along the flow: pixel (x, y) is written to
    (x + flow_x, y + flow_y). Flow is source->target displacement, so this pushes the SOURCE
    image into the TARGET frame (img1 warped by fwd flow ~= img2). Collisions are last-write-wins
    and unmapped pixels stay black (the holes inherent to forward warping).

    Args:
        image: np.ndarray (H, W, 3) in [0, 1] — the image being warped (source).
        flow:  np.ndarray (H, W, 2) — source->target displacement.
        mask:  optional (H, W, 1) source mask; only mapped pixels with mask>0.5 are scattered.
    Returns:
        np.ndarray (H, W, 3) in [0, 1], the source scattered into the target frame.
    """
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    x_dst = np.round(x + flow[..., 0]).astype(np.int64)
    y_dst = np.round(y + flow[..., 1]).astype(np.int64)

    valid = (
        np.isfinite(flow[..., 0])
        & np.isfinite(flow[..., 1])
        & (x_dst >= 0)
        & (x_dst < width)
        & (y_dst >= 0)
        & (y_dst < height)
    )
    if mask is not None:
        valid = valid & (mask[..., 0] > 0.5)

    warped = np.zeros_like(image)
    warped[y_dst[valid], x_dst[valid]] = image[y[valid], x[valid]]
    return warped


def visualize_flow(flow, flow_scale):
    """
    Visualize optical flow with direction modulating color and magnitude modulating saturation in HSV color space.

    Args:
        flow (np.ndarray): Flow array of shape (H, W, 2), where the first dimension
                           represents (flow_x, flow_y).
        flow_scale (float): The scaling factor for the magnitude of the flow.

    Returns:
        np.ndarray: An RGB image visualizing the flow.
    """
    # Convert CHW to HWC
    flow_hwc = flow

    # Compute the magnitude and angle of the flow
    magnitude = np.sqrt(np.square(flow_hwc[..., 0]) + np.square(flow_hwc[..., 1]))
    angle = np.arctan2(flow_hwc[..., 1], flow_hwc[..., 0])  # Angle in radians (-pi, pi)

    # Normalize the magnitude with the provided flow scale
    magnitude = magnitude / flow_scale
    magnitude = np.clip(magnitude, 0, 1)  # Clip values to [0, 1] for saturation

    # Convert angle from radians to degrees (used for color hue in HSV)
    angle_deg = np.degrees(angle) % 360  # Convert angle to [0, 360] degrees

    # Create an HSV image: hue is based on angle, saturation on magnitude, and value is always 1
    hsv_image = np.zeros((flow_hwc.shape[0], flow_hwc.shape[1], 3), dtype=np.uint8)
    hsv_image[..., 0] = angle_deg / 2  # OpenCV expects hue in range [0, 180]
    hsv_image[..., 1] = magnitude * 255  # Saturation in range [0, 255]
    hsv_image[..., 2] = 255  # Value always max (brightest)

    # Convert HSV image to RGB using OpenCV
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return rgb_image

def type_conversion(img, dtype=np.uint8, order="HWC", format="numpy", channel=3):
    if len(img.shape) == 2:
        img = img[..., None]

    # type conversion
    if format == "numpy":
        if isinstance(img, torch.Tensor):
            if img.dtype == torch.bfloat16:
                img = img.float()
            img = img.detach().cpu().numpy()
        elif isinstance(img, np.ndarray):
            img = img
    elif format == "torch":
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        elif isinstance(img, torch.Tensor):
            img = img.detach()
    else:
        raise ValueError("Invalid format, should be 'numpy' or 'torch'")

    # channel order conversion
    if order == "CHW":
        if img.shape[-1] == channel:
            img = img.transpose(2, 0, 1)

        assert img.shape[0] == channel
    elif order == "HWC":
        if img.shape[0] == channel:
            img = img.transpose(1, 2, 0)

        assert img.shape[2] == channel
    else:
        raise ValueError("Invalid order, should be 'HWC' or 'CHW'")

    # range and dtype conversion
    if img.max() <= 1.0:
        if isinstance(img, torch.Tensor):
            img = (img * 255).to(dtype)
        else:
            img = (img * 255).astype(dtype)
    else:
        if isinstance(img, torch.Tensor):
            img = img.to(dtype)
        else:
            img = img.astype(dtype)

    return img


class VisualizerBase:
    def __init__(self, phase: str = "train"):
        pass

    def visualize(
        self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        raise NotImplementedError


class RGBVisualizer(VisualizerBase):
    def __init__(self, field_name: str):
        self.field_name = field_name

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        image_source = None
        data_norm_type = None
        if self.field_name == "batch/img1":
            image_source = batch_data[0]["img"][batch_idx]
            data_norm_type = batch_data[0]["data_norm_type"]
        elif self.field_name == "batch/img2":
            image_source = batch_data[1]["img"][batch_idx]
            data_norm_type = batch_data[1]["data_norm_type"]
        else:
            return None

        # un-normalize
        if data_norm_type is not None:
            image_normalization = IMAGE_NORMALIZATION_DICT[data_norm_type]
            img_mean = image_normalization.mean
            img_std = image_normalization.std
            image_source = image_source * img_std.view(3, 1, 1).to(image_source.device) + img_mean.view(3, 1, 1).to(
                image_source.device
            )

        return type_conversion(image_source, dtype=np.uint8, order="HWC", format="numpy", channel=3)


class FlowVisualizer(VisualizerBase):
    def __init__(self, field_name: str, flow_scale: float = 25):
        self.field_name = field_name
        self.flow_scale = flow_scale

    def visualize(self, batch_idx: int, model_result: Dict[str, Any], batch_data: Dict[str, Any]) -> np.ndarray:
        flow_source = None
        if self.field_name == "batch/flow_fwd":
            flow_source = batch_data[0]["flow"][batch_idx]
        elif self.field_name == "batch/flow_bwd":
            flow_source = batch_data[1]["flow"][batch_idx]
        elif self.field_name == "result/flow_fwd":
            flow_source = model_result.flow.flow_output[batch_idx].detach()
        elif self.field_name == "result/flow_bwd":
            B = batch_data[0]["imgs"].shape[0]
            flow_source = model_result.flow.flow_output[B + batch_idx].detach()
        else:
            return None

        flow_numpy = type_conversion(flow_source, dtype=np.float32, order="HWC", format="numpy", channel=2)
        flow_viz = visualize_flow(flow_numpy, flow_scale=self.flow_scale)
        return flow_viz


class MaskVisualizer(VisualizerBase):
    def __init__(self, field_name: str):
        self.field_name = field_name

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        mask_source = None
        if self.field_name == "result/occlusion_fwd":
            mask_source = model_result.covisibility.mask[batch_idx].detach()
        elif self.field_name == "result/occlusion_bwd":
            B = batch_data[0]["imgs"].shape[0]
            mask_source = model_result.covisibility.mask[B + batch_idx].detach()
        elif self.field_name == "result/inlier_mask":
            mask_source = model_result.inlier_mask.mask[batch_idx].detach()
        elif self.field_name == "batch/occlusion_fwd":
            mask_source = batch_data[0]["non_occluded_mask"][batch_idx]
        elif self.field_name == "batch/occlusion_bwd":
            mask_source = batch_data[1]["non_occluded_mask"][B + batch_idx]
        elif self.field_name == "batch/fov_mask_fwd":
            mask_source = batch_data[0]["fov_mask"][batch_idx]
        elif self.field_name == "batch/fov_mask_bwd":
            mask_source = batch_data[1]["fov_mask"][B + batch_idx]
        else:
            return None

        mask_numpy = type_conversion(mask_source, dtype=np.uint8, order="HWC", format="numpy", channel=1)

        # convert to 3 channels
        mask_numpy = np.repeat(mask_numpy, 3, axis=-1)

        return mask_numpy


class MaskedFlowVisualizer(VisualizerBase):
    def __init__(self, flow_field: str, mask_field: str, flow_scale: float = 25):
        self.flow_field = flow_field
        self.mask_field = mask_field
        self.flow_scale = flow_scale

        self.flow_viz = FlowVisualizer(flow_field, flow_scale)
        self.mask_viz = MaskVisualizer(mask_field)

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        flow_viz = self.flow_viz.visualize(batch_idx, model_result, batch_data)
        mask_viz = self.mask_viz.visualize(batch_idx, model_result, batch_data)

        if flow_viz is None or mask_viz is None:
            return None

        return (flow_viz * (mask_viz.astype(np.float32) / 255)).astype(np.uint8)


class WarpVisualizer(VisualizerBase):
    """Warp the target image (img2) back into the source frame (img1) with a flow field, via
    warp_image_with_flow. With an accurate flow, warped(img2) ≈ img1, so this shows the
    correspondence as an actual reconstructed image. Use flow_field='result/flow_fwd' for the
    PREDICTED warp, 'batch/flow_fwd' for the GT warp. Optionally black-out non-covisible pixels
    via mask_field (e.g. batch/occlusion_fwd)."""

    def __init__(self, flow_field: str = "result/flow_fwd", mask_field: Optional[str] = None,
                 mode: str = "inverse"):
        assert mode in ("inverse", "forward"), f"WarpVisualizer mode must be inverse|forward, got {mode}"
        self.flow_field = flow_field
        self.mask_field = mask_field
        self.mode = mode  # 'inverse': gather img2 into img1 frame (~=img1); 'forward': scatter img1 into img2 frame (~=img2)
        self.src_viz = RGBVisualizer("batch/img1")
        self.tgt_viz = RGBVisualizer("batch/img2")
        self.mask_viz = MaskVisualizer(mask_field) if mask_field is not None else None

    def visualize(
        self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        source = self.src_viz.visualize(batch_idx, model_result, batch_data)  # uint8 HWC, img1
        target = self.tgt_viz.visualize(batch_idx, model_result, batch_data)  # uint8 HWC, img2
        if source is None or target is None:
            return None

        if self.flow_field == "result/flow_fwd":
            flow_source = model_result.flow.flow_output[batch_idx].detach()
        elif self.flow_field == "batch/flow_fwd":
            flow_source = batch_data[0]["flow"][batch_idx]
        else:
            return None

        # raw pixel displacements (H, W, 2); avoid type_conversion's <=1.0 auto-scaling
        flow_np = np.nan_to_num(flow_source.float().cpu().numpy().transpose(1, 2, 0))

        mask_np = None
        if self.mask_viz is not None:
            mask_img = self.mask_viz.visualize(batch_idx, model_result, batch_data)  # uint8 HWC, 3ch
            if mask_img is not None:
                mask_np = mask_img[..., :1].astype(np.float32) / 255.0

        if self.mode == "inverse":
            # gather img2 into img1's frame; reconstructs img1
            warped = warp_image_with_flow(
                source.astype(np.float32) / 255.0, mask_np, target.astype(np.float32) / 255.0, flow_np
            )
        else:
            # scatter img1 into img2's frame using the flow; reconstructs img2
            warped = forward_warp_image(source.astype(np.float32) / 255.0, flow_np, mask_np)
        return np.clip(warped * 255.0, 0, 255).astype(np.uint8)


class MaskedFlowLossVisualizer(VisualizerBase):
    def __init__(self, flow_field: str, mask_field: str, flow_scale: float = 25, loss_config: Dict[str, Any] = {}):
        self.flow_field = flow_field
        self.mask_field = mask_field
        self.flow_scale = flow_scale

        self.flow_viz = FlowVisualizer(flow_field, flow_scale)
        self.mask_viz = MaskVisualizer(mask_field)

        self.loss = get_loss(loss_config["class"], **loss_config["kwargs"])

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        flow_viz = self.flow_viz.visualize(batch_idx, model_result, batch_data)
        mask_viz = self.mask_viz.visualize(batch_idx, model_result, batch_data)

        enabled, loss, extra_info = self.loss.compute_loss(batch_data, model_result)
        epe = extra_info["flow_epe@non_occluded_mask"]

        if flow_viz is None or mask_viz is None:
            return None

        img = (flow_viz * (mask_viz.astype(np.float32) / 255)).astype(np.uint8)

        # put epe as text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        font_color = (255, 255, 255)
        font_position = (10, 20)
        cv2.putText(img, f"EPE: {epe:.2f}", font_position, font, font_scale, font_color, font_thickness)

        return img


class ErrorVisualizer(VisualizerBase):

    def __init__(self, range="occlusion", target: str = "flow_output"):
        self.range = range
        self.target = target
        self.cmap = plt.get_cmap("coolwarm")

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        if self.range == "occlusion":
            mask_source = batch_data[0]["non_occluded_mask"][batch_idx]
        elif self.range == "fov_mask":
            mask_source = batch_data[0]["fov_mask"][batch_idx]
        else:
            return None

        if self.target == "flow_output":
            result_batch = model_result.flow.flow_output.detach()
        elif self.target == "regression_flow_output":
            result_batch = model_result.classification_refinement.regression_flow_output.detach()
        else:
            raise ValueError("Invalid target selection")

        flow_output = result_batch[batch_idx]
        flow_gt = batch_data[0]["flow"][batch_idx]

        reference_flow = flow_gt.unsqueeze(0)
        reference_mask = mask_source.unsqueeze(0)
        reference_mask_float = reference_mask.float()
        flow_output = result_batch[batch_idx].unsqueeze(0)

        reference_mask = reference_mask & (~reference_flow.isnan().any(dim=1))
        reference_mask = reference_mask & (torch.abs(reference_flow).sum(dim=1) < 2048)

        reference_flow[reference_flow.isnan()] = (
            0  # set nan to 0, will be masked out later by reference_mask or reference_mask_float
        )

        flow_output[torch.isnan(flow_output)] = 0  # set nan to 0, the canonical guess for somehow invalid flow
        reference_flow[torch.abs(reference_flow) > 2048] = 0  # for kitti training only to transmit invalidity

        flow_epe = torch.linalg.norm(flow_output - reference_flow, dim=1)
        flow_epe = flow_epe * reference_mask_float
        epe = flow_epe.sum() / (reference_mask_float.sum() + 1e-6)

        error = flow_epe[0].detach().cpu().numpy()
        vmax = error.max()

        normalized_data = np.clip(error / vmax, 0, 1)

        # Apply the viridis colormap
        colored_data = self.cmap(normalized_data)
        colored_data[~mask_source.cpu().numpy()] = 255  # set occluded area to black

        # Convert to uint8 for visualization
        vis_data = (colored_data[..., :3] * 255).astype(np.uint8)

        # put epe as text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        font_color = (255, 255, 255)
        font_position = (10, 20)
        cv2.putText(vis_data, f"EPE: {epe:.2f}", font_position, font, font_scale, font_color, font_thickness)

        return vis_data


class ErrorOutlierVisualizer(VisualizerBase):
    def __init__(self, range="occlusion", target: str = "flow_output"):
        self.range = range
        self.target = target

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        if self.range == "occlusion":
            mask_source = batch_data[0]["non_occluded_mask"][batch_idx]
        elif self.range == "fov_mask":
            mask_source = batch_data[0]["fov_mask"][batch_idx]
        else:
            return None

        if self.target == "flow_output":
            result_batch = model_result.flow.flow_output.detach()
        elif self.target == "regression_flow_output":
            result_batch = model_result.classification_refinement.regression_flow_output.detach()
        else:
            raise ValueError("Invalid target selection")

        flow_output = result_batch[batch_idx]
        flow_gt = batch_data[0]["flow"][batch_idx]

        reference_flow = flow_gt.unsqueeze(0)
        reference_mask = mask_source.unsqueeze(0)
        reference_mask_float = reference_mask.float()
        flow_output = result_batch[batch_idx].unsqueeze(0)

        reference_mask = reference_mask & (~reference_flow.isnan().any(dim=1))
        reference_mask = reference_mask & (torch.abs(reference_flow).sum(dim=1) < 2048)

        reference_flow[reference_flow.isnan()] = 0
        flow_output[torch.isnan(flow_output)] = 0
        reference_flow[torch.abs(reference_flow) > 2048] = 0

        flow_epe = torch.linalg.norm(flow_output - reference_flow, dim=1)
        flow_epe = flow_epe * reference_mask_float
        epe = flow_epe.sum() / (reference_mask_float.sum() + 1e-6)

        error = flow_epe[0].detach().cpu().numpy()

        # Define thresholds and corresponding colors
        thresholds = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        colors = np.array(
            [
                [255, 0, 0],  # Red
                [255, 0, 0],  # Red
                [255, 165, 0],  # Orange
                [255, 255, 0],  # Yellow
                [0, 255, 0],  # Green
                [0, 255, 255],  # Cyan
                [0, 0, 255],  # Blue
                [128, 0, 128],  # Purple
            ],
            dtype=np.uint8,
        )[::-1, :]

        # Use digitize to find indices for color mapping
        indices = np.digitize(error, thresholds, right=True)
        vis_data = colors[indices]

        vis_data[~mask_source.cpu().numpy()] = 255  # set occluded area to white

        # Put EPE text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        font_color = (255, 255, 255)
        font_position = (10, 20)
        cv2.putText(vis_data, f"EPE: {epe:.2f}", font_position, font, font_scale, font_color, font_thickness)

        return vis_data


class RefinementFeatureVisualizer(VisualizerBase):

    def __init__(self, selection: str = "feature0"):
        self.selection = selection

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:

        feature0 = model_result.classification_refinement.feature_map_0[batch_idx].detach().clone()
        feature1 = model_result.classification_refinement.feature_map_1[batch_idx].detach().clone()

        # Assume feature0, feature1 are [C, H, W]
        C, H, W = feature0.shape

        # Reshape to [H*W, C]
        f0 = feature0.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
        f1 = feature1.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]

        # Stack into [2*H*W, C]
        all_feats = torch.cat([f0, f1], dim=0)  # [2*H*W, C]
        all_feats = all_feats.chunk(2, dim=0)

        all_feats = all_feats[0] if self.selection == "feature0" else all_feats[1]

        try:
            # Apply PCA
            pca = PCA(n_components=3)
            projected = pca.fit_transform(all_feats.detach().float().cpu().numpy())  # [2*H*W, 3]

            # Split back
            rgb = projected.reshape(H, W, 3)

            # Normalize for visualization
            def normalize_img(img):
                img -= img.min()
                img /= img.max()
                return img

            rgb = normalize_img(rgb)
            rgb = (rgb * 255.0).astype(np.uint8)
        except:
            return np.zeros((H, W, 3)).astype(np.uint8)

        return rgb


class ConcatVisualizer(VisualizerBase):
    def __init__(self, visualizers: List[VisualizerBase], arrangement: Optional[str] = None):
        self.visualizers = visualizers
        self.arrangement = arrangement

    def visualize(
        self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        images = []
        for visualizer in self.visualizers:
            image = visualizer.visualize(batch_idx, model_result, batch_data)
            if image is not None:
                images.append(image)

        if len(images) == 0:
            return None

        if self.arrangement is None:
            # Default: horizontal concat
            return np.concatenate(images, axis=1)

        try:
            rows, cols = map(int, self.arrangement.lower().split("x"))
        except Exception as e:
            raise ValueError(f"Invalid arrangement format '{self.arrangement}', expected format like '2x3'.") from e

        if rows * cols < len(images):
            raise ValueError(f"Arrangement {rows}x{cols} can't fit {len(images)} images.")

        # Pad with blank images if needed
        h, w, c = images[0].shape
        blank = np.zeros_like(images[0])
        images += [blank] * (rows * cols - len(images))

        # Build the grid
        grid_rows = []
        for i in range(rows):
            row = np.concatenate(images[i * cols : (i + 1) * cols], axis=1)
            grid_rows.append(row)

        return np.concatenate(grid_rows, axis=0)


class CovarianceVisualizer(VisualizerBase):
    def __init__(self, axis: str, max_cov_normalizer: Optional[float] = None):
        self.axis = axis
        self.max_cov_normalizer = max_cov_normalizer
        self.cmap = plt.get_cmap("viridis")

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        covariance_matrix = model_result.flow.flow_covariance

        # Select the appropriate axis
        if self.axis == "xx":
            cov_slice = covariance_matrix[batch_idx][0]
        elif self.axis == "yy":
            cov_slice = covariance_matrix[batch_idx][1]
        else:
            return None

        cov_slice = cov_slice.detach().cpu().numpy()

        # Normalize the data
        if self.max_cov_normalizer is not None:
            vmax = self.max_cov_normalizer
        else:
            vmax = cov_slice.max()

        normalized_data = np.clip(cov_slice / vmax, 0, 1)

        # Apply the viridis colormap
        colored_data = self.cmap(normalized_data)

        # Convert to uint8 for visualization
        vis_data = (colored_data[..., :3] * 255).astype(np.uint8)

        return vis_data


class CovarianceRatioVisualizer(VisualizerBase):
    def __init__(self, axis: str, flow_target: str = "flow_output", vmin: float = 0.0, vmax: float = 5.0):
        self.axis = axis
        self.flow_target = flow_target
        self.cmap = plt.get_cmap("seismic")

        self.norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        covariance_matrix = model_result.flow.flow_covariance

        flow_source = (
            model_result.flow.flow_output
            if self.flow_target == "flow_output"
            else model_result.classification_refinement.regression_flow_output
        )

        # Select the appropriate axis
        if self.axis == "xx":
            cov_slice = covariance_matrix[batch_idx][0]
            flow_pred_slice = flow_source[batch_idx][0]
            flow_gt_slice = batch_data[0]["flow"][batch_idx, 0]
        elif self.axis == "yy":
            cov_slice = covariance_matrix[batch_idx][1]
            flow_pred_slice = flow_source[batch_idx][1]
            flow_gt_slice = batch_data[0]["flow"][batch_idx, 1]
        else:
            return None

        std_slice = torch.sqrt(cov_slice.detach() + 1e-4).cpu().numpy()
        error_slice = torch.abs(flow_pred_slice.detach() - flow_gt_slice.detach()).cpu().numpy()

        flow_error_ratio = error_slice / std_slice

        # Apply the viridis colormap
        colored_data = self.cmap(self.norm(flow_error_ratio))

        # Convert to uint8 for visualization
        vis_data = (colored_data[..., :3] * 255).astype(np.uint8)

        return vis_data


class ConfidenceNMSVisualizer(VisualizerBase):
    def __init__(self, visualizing="confidence"):
        self.cmap = plt.get_cmap("jet")
        self.visualizing = visualizing

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:
        covariance_matrix = model_result.flow.flow_covariance

        confidence = 1.0 / (1e-4 + covariance_matrix[batch_idx][0])
        nms_confidence = 1.0 / (1e-4 + covariance_matrix[batch_idx][1])
        confidence = confidence.detach().cpu().numpy()
        nms_confidence = nms_confidence.detach().cpu().numpy()

        if self.visualizing == "nms_confidence":
            # nms = neighborhood_softmax_memory_efficient(torch.from_numpy(confidence).unsqueeze(0), N=7, temp=0.1).squeeze(0).numpy()
            nms = (
                neighborhood_softmax_stable_batched(
                    torch.from_numpy(nms_confidence).unsqueeze(0).unsqueeze(0), N=7, temp=0.1
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )
            nms_confidence = nms * confidence
            confidence = nms_confidence

        vmax = confidence.max()

        normalized_data = np.clip(confidence / vmax, 0, 1)

        # Apply the viridis colormap
        colored_data = self.cmap(normalized_data)

        # Convert to uint8 for visualization
        vis_data = (colored_data[..., :3] * 255).astype(np.uint8)

        return vis_data


def neighborhood_softmax_stable_batched(confidence, N=3, temp=0.1):
    """
    Memory-efficient, numerically stable softmax over NxN neighborhoods for batched inputs.

    Args:
        confidence (torch.Tensor): Input tensor of shape (B, C, H, W)
        N (int): Neighborhood size (must be odd)
        temp (float): Temperature parameter for softmax sharpening

    Returns:
        torch.Tensor: Output tensor of same shape as input
    """
    assert N % 2 == 1, "N must be an odd integer"
    B, C, H, W = confidence.shape
    pad_size = N // 2

    # 1. Compute local maxima for numerical stability
    local_max = torch.nn.functional.max_pool2d(confidence, kernel_size=N, stride=1, padding=pad_size)

    # 2. Stabilize and apply temperature scaling
    stabilized = (confidence - local_max) / temp

    # 3. Compute exponentials
    exp_vals = torch.exp(stabilized)

    # 4. Create convolution kernel and sum neighborhoods
    kernel = torch.ones((1, 1, N, N), dtype=exp_vals.dtype, device=exp_vals.device)

    # Channel-wise summation (groups=C ensures per-channel processing)
    sum_exp = torch.nn.functional.conv2d(
        exp_vals, kernel, padding=pad_size, groups=C  # Critical for correct channel handling
    )

    # 5. Compute final softmax probabilities
    return exp_vals / (sum_exp + 1e-6)


class ConfidenceVisualizer(VisualizerBase):
    def __init__(self, visualizing="keypoint_confidence"):
        self.cmap = plt.get_cmap("jet")
        self.visualizing = visualizing

    def visualize(self, batch_idx: int, model_result: UFMOutputInterface, batch_data: Dict[str, Any]) -> np.ndarray:

        if self.visualizing == "keypoint_confidence":
            confidence = model_result.keypoint_confidence[batch_idx]
        else:
            return None

        confidence = confidence.detach().cpu().numpy()

        vmax = confidence.max()
        vmin = confidence.min()
        normalized_data = np.clip((confidence - vmin) / (vmax - vmin), 0, 1)

        # Apply the viridis colormap
        colored_data = self.cmap(normalized_data)

        # Convert to uint8 for visualization
        vis_data = (colored_data[..., :3] * 255).astype(np.uint8)

        return vis_data


def get_visualizer(viz_config: Dict[str, Any]) -> VisualizerBase:
    class_name = viz_config["class"]
    kwargs = viz_config["kwargs"]

    visualizer = None
    if class_name == "RGBVisualizer":
        visualizer = RGBVisualizer(**kwargs)
    elif class_name == "FlowVisualizer":
        visualizer = FlowVisualizer(**kwargs)
    elif class_name == "MaskVisualizer":
        visualizer = MaskVisualizer(**kwargs)
    elif class_name == "ConcatVisualizer":
        visualizer = ConcatVisualizer(
            [get_visualizer(v) for v in kwargs["visualizers"]],
            arrangement=kwargs["arrangement"] if "arrangement" in kwargs else None,
        )
    elif class_name == "MaskedFlowVisualizer":
        visualizer = MaskedFlowVisualizer(**kwargs)
    elif class_name == "WarpVisualizer":
        visualizer = WarpVisualizer(**kwargs)
    elif class_name == "MaskedFlowLossVisualizer":
        visualizer = MaskedFlowLossVisualizer(**kwargs)
    elif class_name == "CovarianceVisualizer":
        visualizer = CovarianceVisualizer(**kwargs)
    elif class_name == "ConfidenceNMSVisualizer":
        visualizer = ConfidenceNMSVisualizer(**kwargs)
    elif class_name == "ConfidenceVisualizer":
        visualizer = ConfidenceVisualizer(**kwargs)
    elif class_name == "ErrorVisualizer":
        visualizer = ErrorVisualizer(**kwargs)
    elif class_name == "ErrorOutlierVisualizer":
        visualizer = ErrorOutlierVisualizer(**kwargs)
    elif class_name == "RefinementFeatureVisualizer":
        visualizer = RefinementFeatureVisualizer(**kwargs)
    elif class_name == "CovarianceRatioVisualizer":
        visualizer = CovarianceRatioVisualizer(**kwargs)
    else:
        raise ValueError(f"Invalid visualizer class: {class_name}")

    return visualizer
