import argparse
import copy
import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules
from mmengine.config import Config, DictAction
from mmengine.structures import InstanceData
from mmengine.utils import import_modules_from_strings


SCANNET_CLASSES = (
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain",
    "toilet", "sink", "bathtub", "garbage bin")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Grounding DINO inference on a Dudet dataset")
    parser.add_argument("dino_config", help="Grounding DINO config file")
    parser.add_argument("dino_checkpoint", help="Grounding DINO checkpoint")
    parser.add_argument("dataset_config", help="Dudet dataset config file")
    parser.add_argument("--out-dir", required=True, help="directory for JSON predictions")
    parser.add_argument("--device", default="cuda:0", help="inference device")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--max-per-view", type=int, default=100)
    parser.add_argument(
        "--project-gt", action="store_true",
        help="project 3D ground-truth boxes into every view")
    parser.add_argument(
        "--visualize", action="store_true", help="save rendered detections")
    parser.add_argument(
        "--vis-dir", help="directory for rendered detection images")
    parser.add_argument(
        "--text-prompt",
        default=" . ".join(SCANNET_CLASSES),
        help="Grounding DINO text prompt")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override settings in the Dudet dataset config")
    return parser.parse_args()


def build_dataset(config_path, cfg_options):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    custom_imports = cfg.get("custom_imports")
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)

    dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dataset_cfg.pipeline = []
    dataset_cfg.test_mode = True
    return DATASETS.build(dataset_cfg)


def instances_to_dict(instances, classes, score_thr, max_per_view):
    instances = instances[instances.scores >= score_thr]
    if len(instances) > max_per_view:
        keep = instances.scores.topk(max_per_view).indices
        instances = instances[keep]
    labels = instances.labels.detach().cpu().tolist()
    return dict(
        bboxes=instances.bboxes.detach().cpu().tolist(),
        scores=instances.scores.detach().cpu().tolist(),
        labels=labels,
        label_names=[classes[label] for label in labels])


def project_gt_to_2d(data_info, view_index, image_shape):
    ann_info = data_info.get("ann_info")
    if ann_info is None:
        return dict(bboxes=[], labels=[], label_names=[])

    corners = ann_info["gt_bboxes_3d"].corners.detach().cpu().numpy()
    labels = ann_info["gt_labels_3d"].tolist()
    extrinsic = np.asarray(data_info["lidar2img"]["extrinsic"][view_index])
    intrinsics = np.asarray(data_info["lidar2img"]["intrinsic"])
    intrinsic = intrinsics[view_index] if intrinsics.ndim == 3 else intrinsics
    if intrinsic.shape == (3, 3):
        intrinsic_4x4 = np.eye(4, dtype=intrinsic.dtype)
        intrinsic_4x4[:3, :3] = intrinsic
        intrinsic = intrinsic_4x4
    height, width = image_shape[:2]
    projected_bboxes = []
    projected_labels = []
    for box_corners, label in zip(corners, labels):
        corners_hom = np.concatenate(
            [box_corners, np.ones((len(box_corners), 1))], axis=1)
        corners_cam = corners_hom @ extrinsic.T
        valid = corners_cam[:, 2] > 1e-5
        if not valid.any():
            continue
        pixels = corners_cam[valid] @ intrinsic.T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        x1, y1 = pixels.min(axis=0)
        x2, y2 = pixels.max(axis=0)
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, width - 1.0), min(y2, height - 1.0)
        if x2 <= x1 or y2 <= y1:
            continue
        projected_bboxes.append([float(x1), float(y1), float(x2), float(y2)])
        projected_labels.append(int(label))

    return dict(
        bboxes=projected_bboxes,
        labels=projected_labels,
        label_names=[SCANNET_CLASSES[label] for label in projected_labels])


def gt_to_data_sample(gt_2d):
    data_sample = DetDataSample()
    instances = InstanceData()
    instances.bboxes = torch.tensor(gt_2d["bboxes"], dtype=torch.float32).reshape(-1, 4)
    instances.labels = torch.tensor(gt_2d["labels"], dtype=torch.long)
    instances.scores = torch.ones(len(instances.labels), dtype=torch.float32)
    data_sample.pred_instances = instances
    return data_sample


def main():
    args = parse_args()
    register_all_modules(init_default_scope=False)
    dataset = build_dataset(args.dataset_config, args.cfg_options)
    model = init_detector(
        args.dino_config, args.dino_checkpoint, device=args.device)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = None
    if args.visualize:
        visualizer = VISUALIZERS.build(copy.deepcopy(model.cfg.visualizer))
        visualizer.dataset_meta = dict(classes=SCANNET_CLASSES)
        vis_dir = Path(args.vis_dir) if args.vis_dir else output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    for sample_index in range(len(dataset)):
        data_info = dataset.get_data_info(sample_index)
        image_paths = [item["filename"] for item in data_info["img_info"]]
        sample_id = str(data_info.get("sample_idx", sample_index))
        views = []
        for view_index, image_path in enumerate(image_paths):
            image = mmcv.imread(image_path, channel_order="rgb")
            result = inference_detector(
                model,
                image_path,
                text_prompt=args.text_prompt,
                custom_entities=True)
            if visualizer is not None:
                visualizer.add_datasample(
                    name=f"{sample_id}_{view_index:03d}",
                    image=image,
                    data_sample=result,
                    draw_gt=False,
                    show=False,
                    pred_score_thr=args.score_thr,
                    out_file=str(vis_dir / f"{sample_id}_{view_index:03d}.png"))
            gt_2d = None
            if args.project_gt:
                gt_2d = project_gt_to_2d(data_info, view_index, image.shape)
                if visualizer is not None:
                    visualizer.add_datasample(
                        name=f"{sample_id}_{view_index:03d}_gt",
                        image=image,
                        data_sample=gt_to_data_sample(gt_2d),
                        draw_gt=False,
                        show=False,
                        pred_score_thr=0.0,
                        out_file=str(vis_dir / f"{sample_id}_{view_index:03d}_gt.png"))
            print(result)
            view_result = dict(
                image_path=image_path,
                **instances_to_dict(
                    result.pred_instances,
                    SCANNET_CLASSES,
                    args.score_thr,
                    args.max_per_view))
            if gt_2d is not None:
                view_result["gt_2d"] = gt_2d
            views.append(view_result)

        output_path = output_dir / f"{sample_id}.json"
        output_path.write_text(json.dumps(dict(views=views)))


if __name__ == "__main__":
    main()
