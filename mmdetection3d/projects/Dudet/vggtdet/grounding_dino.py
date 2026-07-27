from typing import Sequence

import torch
import torch.nn as nn
from mmcv.ops import nms
from mmcv.transforms import Compose
from mmdet.registry import MODELS
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import DefaultScope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData

from projects.Dudet.vggtdet.device import get_device


SCANNET_CLASSES = (
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain",
    "toilet", "sink", "bathtub", "garbage bin")


class GroundingDINO2DDetector(nn.Module):
    def __init__(self,
                 config: str,
                 checkpoint: str,
                 score_thr: float = 0.25,
                 nms_iou_thr: float = 0.5,
                 max_per_view: int = 100,
                 inference_batch_size: int = 1,
                 use_grounding_dino: bool = True,
                 classes: Sequence[str] = None,
                 device=None) -> None:
        super().__init__()
        device = get_device() if device is None else device
        self.use_grounding_dino = use_grounding_dino
        self.detector = None
        self.test_pipeline = None
        if self.use_grounding_dino:
            with DefaultScope.overwrite_default_scope("mmdet"):
                self.detector = self._init_detector(config, checkpoint, device)
                pipeline_cfg = get_test_pipeline_cfg(self.detector.cfg.copy())
                self.test_pipeline = Compose(pipeline_cfg)
        self.score_thr = score_thr
        self.nms_iou_thr = nms_iou_thr
        self.max_per_view = max_per_view
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be at least 1")
        self.inference_batch_size = inference_batch_size
        self.classes = tuple(classes) if classes is not None else SCANNET_CLASSES
        self.text_prompt = ". ".join(self.classes) + "."
        if self.detector is not None:
            self.detector.requires_grad_(False)
            self.detector.eval()

    @staticmethod
    def _init_detector(config, checkpoint, device):
        config = Config.fromfile(config)
        backbone = config.model.get("backbone")
        if backbone is not None and "init_cfg" in backbone:
            backbone.init_cfg = None
        detector = revert_sync_batchnorm(MODELS.build(config.model))
        load_checkpoint(detector, checkpoint, map_location="cpu")
        detector.cfg = config
        detector.to(device)
        detector.eval()
        return detector

    def train(self, mode: bool = True):
        super().train(False)
        if self.detector is not None:
            self.detector.eval()
        return self

    @staticmethod
    def _project_gt_instances(data_sample, view_index, image_shape, device):
        gt_instances = data_sample.gt_instances_3d
        corners = gt_instances.bboxes_3d.corners.to(device=device, dtype=torch.float32)
        labels = gt_instances.labels_3d.to(device=device, dtype=torch.long)
        if len(corners) == 0:
            return InstanceData(
                bboxes=torch.empty((0, 4), device=device),
                labels=torch.empty((0,), dtype=torch.long, device=device),
                scores=torch.empty((0,), device=device))

        lidar2img = data_sample.metainfo["lidar2img"]
        extrinsic = torch.as_tensor(
            lidar2img["extrinsic"][view_index], dtype=torch.float32,
            device=device)
        intrinsics = torch.as_tensor(
            lidar2img["intrinsic"], dtype=torch.float32, device=device)
        intrinsic = intrinsics[view_index] if intrinsics.ndim == 3 else intrinsics
        if intrinsic.shape == (3, 3):
            intrinsic_4x4 = torch.eye(4, dtype=intrinsic.dtype, device=device)
            intrinsic_4x4[:3, :3] = intrinsic
            intrinsic = intrinsic_4x4

        corners_hom = torch.cat(
            [corners, torch.ones_like(corners[..., :1])], dim=-1)
        corners_cam = corners_hom @ extrinsic.T
        valid = corners_cam[..., 2] > 1e-5
        pixels = corners_cam @ intrinsic.T
        pixels = pixels[..., :2] / pixels[..., 2:3].clamp_min(1e-5)

        height, width = image_shape[:2]
        bboxes = []
        projected_labels = []
        for box_pixels, box_valid, label in zip(pixels, valid, labels):
            box_pixels = box_pixels[box_valid]
            if len(box_pixels) == 0:
                continue
            xy_min = box_pixels.min(dim=0).values.clamp(min=0)
            xy_max = box_pixels.max(dim=0).values
            xy_max[0].clamp_(max=width - 1)
            xy_max[1].clamp_(max=height - 1)
            if (xy_max <= xy_min).any():
                continue
            bboxes.append(torch.cat([xy_min, xy_max]))
            projected_labels.append(label)

        if not bboxes:
            return InstanceData(
                bboxes=torch.empty((0, 4), device=device),
                labels=torch.empty((0,), dtype=torch.long, device=device),
                scores=torch.empty((0,), device=device))
        return InstanceData(
            bboxes=torch.stack(bboxes),
            labels=torch.stack(projected_labels),
            scores=torch.ones(len(bboxes), device=device))

    def _merge_and_nms(self, pred_instances, gt_instances):
        pred_instances = pred_instances[pred_instances.scores >= self.score_thr]
        bboxes = torch.cat([pred_instances.bboxes, gt_instances.bboxes])
        scores = torch.cat([pred_instances.scores, gt_instances.scores])
        labels = torch.cat([pred_instances.labels, gt_instances.labels])
        if len(bboxes) == 0:
            return InstanceData(bboxes=bboxes, scores=scores, labels=labels)

        _, keep = nms(bboxes, scores, self.nms_iou_thr)
        if len(keep) > self.max_per_view:
            keep = keep[:self.max_per_view]
        return InstanceData(
            bboxes=bboxes[keep], scores=scores[keep], labels=labels[keep])

    def _infer_views(self, image_paths):
        results = []
        for start in range(0, len(image_paths), self.inference_batch_size):
            batch_inputs = []
            batch_data_samples = []
            for image_path in image_paths[
                    start:start + self.inference_batch_size]:
                data = self.test_pipeline(
                    dict(
                        img_path=image_path,
                        img_id=0,
                        text=self.text_prompt,
                        custom_entities=True))
                batch_inputs.append(data["inputs"])
                batch_data_samples.append(data["data_samples"])
            data = self.detector.data_preprocessor(
                dict(inputs=batch_inputs, data_samples=batch_data_samples),
                training=False)
            results.extend(self.detector.predict(
                data["inputs"], data["data_samples"]))
        return results

    @staticmethod
    def _get_view_shapes(data_sample, num_views):
        image_shapes = data_sample.metainfo["ori_shape"]
        if isinstance(image_shapes, torch.Tensor):
            image_shapes = image_shapes.tolist()
        if len(image_shapes) > 0 and isinstance(image_shapes[0], (int, float)):
            image_shapes = [image_shapes] * num_views
        if len(image_shapes) != num_views:
            raise ValueError("ori_shape must provide one shape for each view")
        return image_shapes

    @torch.no_grad()
    def forward(self, batch_data_samples):
        batch_instances = []
        for data_sample in batch_data_samples:
            image_paths = data_sample.metainfo["img_path"]
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            if self.use_grounding_dino:
                with DefaultScope.overwrite_default_scope("mmdet"):
                    results = self._infer_views(image_paths)
                view_instances = []
                for view_index, result in enumerate(results):
                    gt_instances = self._project_gt_instances(
                        data_sample, view_index,
                        result.metainfo["ori_shape"],
                        result.pred_instances.bboxes.device)
                    view_instances.append(
                        self._merge_and_nms(result.pred_instances, gt_instances))
            else:
                device = data_sample.gt_instances_3d.bboxes_3d.tensor.device
                view_instances = [
                    self._project_gt_instances(
                        data_sample, view_index, image_shape, device)
                    for view_index, image_shape in enumerate(
                        self._get_view_shapes(data_sample, len(image_paths)))
                ]
            batch_instances.append(view_instances)
        return batch_instances
