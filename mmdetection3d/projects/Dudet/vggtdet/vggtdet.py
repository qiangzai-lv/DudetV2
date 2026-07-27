import sys
from pathlib import Path
from numbers import Number
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from projects.Dudet.detr3_models.helpers import GenericMLP
from projects.Dudet.detr3_models.utils.votenet_pc_util import write_ply_rgb
from projects.Dudet.vggtdet.device import autocast, get_device
from projects.Dudet.vggtdet.grounding_dino import GroundingDINO2DDetector
from projects.Dudet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from projects.Dudet.detr3_models.transformer import (TransformerDecoder, TransformerDecoder_Multilevel,
                                                     TransformerDecoderLayer)
_VGGT_OMEGA_ROOT = Path(__file__).resolve().parents[3] / 'vggt-omega'
if str(_VGGT_OMEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_OMEGA_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

device = get_device()


@torch.no_grad()
def unproject_depth_map_to_point_map_torch(
        depth_map: torch.Tensor, extrinsics: torch.Tensor,
        intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject depth maps with OpenCV camera-from-world extrinsics."""
    batch_size, num_views, height, width = depth_map.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth_map.device, dtype=depth_map.dtype),
        torch.arange(width, device=depth_map.device, dtype=depth_map.dtype),
        indexing='ij')
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)
    fx = intrinsics[..., 0, 0].view(batch_size, num_views, 1, 1)
    fy = intrinsics[..., 1, 1].view(batch_size, num_views, 1, 1)
    cx = intrinsics[..., 0, 2].view(batch_size, num_views, 1, 1)
    cy = intrinsics[..., 1, 2].view(batch_size, num_views, 1, 1)
    camera_points = torch.stack(
        [(x - cx) * depth_map / fx, (y - cy) * depth_map / fy, depth_map],
        dim=-1)
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return torch.einsum(
        'bsji,bshwj->bshwi', rotation,
        camera_points - translation[:, :, None, None, :])

class ChannelProjecter(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels//2,
                    kernel_size=1,
                    stride=1,
                    padding=0
                            ),
            nn.GroupNorm(num_groups=1, num_channels=in_channels//2),
            nn.GELU(),
            nn.Conv2d(
                    in_channels=in_channels//2,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                            )
        )
        
        self.res = nn.Sequential(
            nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                            )
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.proj(x) + self.res(x)
        del x
        return res   # [B, D, N, T]
    
@MODELS.register_module()
class VGGTDet(Base3DDetector):
    def __init__(
            self,
            bbox_head: ConfigType,
            two_d_detector: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptConfigType = None,
            decoder_cfg: OptConfigType = None,
            if_learnable_query=True,
            num_queries=128,
            token_dim=1024,
            test_only_last_layer=True,
            if_use_gt_query=False,
            position_embedding="fourier",
            if_mix_precision=False,
            use_multi_layers=False,
            if_simpler_project=False,
            if_use_pred_pc_query=False,
            depth_thres=1000,
            if_task_query=False,
            vggt_omega_checkpoint='/mnt/workspace/pretrain/VGGT-Omega/vggt_omega_1b_512.pt',
            visualize_pred_pointcloud=False,
            pred_pointcloud_path='vis_dir/pred_points',
            query_3d_nms_iou_thr=0.25,
            query_min_points=16,
            keyframe_count=0
            ):
        
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg) 

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.two_d_detector = None
        if two_d_detector and two_d_detector.get("enabled", False):
            self.two_d_detector = GroundingDINO2DDetector(
                config=two_d_detector["config"],
                checkpoint=two_d_detector["checkpoint"],
                score_thr=two_d_detector.get("score_thr", 0.25),
                nms_iou_thr=two_d_detector.get("nms_iou_thr", 0.5),
                max_per_view=two_d_detector.get("max_per_view", 100),
                inference_batch_size=two_d_detector.get(
                    "inference_batch_size", 8),
                use_grounding_dino=two_d_detector.get(
                    "use_grounding_dino", True),
                classes=two_d_detector.get("classes"),
                device=device)
        self.vggt_encoder = VGGTOmega()
        self.vggt_encoder.load_state_dict(
            torch.load(vggt_omega_checkpoint, map_location='cpu'))
        self.vggt_encoder.to(device)

        for param in self.vggt_encoder.parameters():
            param.requires_grad = False

        self.vggt_encoder.eval()

        self.decoder = build_decoder(decoder_cfg, if_multilevel=use_multi_layers)

        if if_simpler_project:
            if use_multi_layers: 
                self.proj_feat_dim0 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim1 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim2 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim3 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            else:
                self.proj_feat_dim = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
        else:
            if use_multi_layers: 
                self.proj_feat_dim0 = ChannelProjecter(in_channels=2048, out_channels=token_dim) #for _ in range(4)]
                self.proj_feat_dim1 = ChannelProjecter(in_channels=2048, out_channels=token_dim) 
                self.proj_feat_dim2 = ChannelProjecter(in_channels=2048, out_channels=token_dim) 
                self.proj_feat_dim3 = ChannelProjecter(in_channels=2048, out_channels=token_dim)
            else:
                self.proj_feat_dim = ChannelProjecter(in_channels=2048, out_channels=token_dim)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


        # self.proj_norm = nn.LayerNorm(token_dim)
        self.num_queries = num_queries
        self.if_learnable_query = if_learnable_query

        if if_learnable_query:
            self.queries = nn.Parameter(torch.Tensor(num_queries, token_dim))
            nn.init.xavier_normal_(self.queries)
        ######### idea 2 ############
        self.if_task_query = if_task_query
        if if_task_query:
            self.task_query = nn.Parameter(torch.Tensor(1, token_dim))
            nn.init.xavier_normal_(self.task_query)
        ######### idea 2 ############
        self.test_only_last_layer = test_only_last_layer

        self.if_use_gt_query = if_use_gt_query
        # assert if_learnable_query is not self.if_use_gt_query

        self.if_use_pred_pc_query = if_use_pred_pc_query
        # assert 
        assert (self.if_use_pred_pc_query + self.if_use_gt_query + self.if_learnable_query) == 1, \
            "Only one of 'if_use_pred_pc_query', 'if_use_gt_query', or 'if_learnable_query' must be True."
        
        if self.if_use_gt_query or self.if_use_pred_pc_query:
            self.pos_embedding = PositionEmbeddingCoordsSine(
                d_pos=token_dim, pos_type=position_embedding, normalize=False
            )
            self.query_projection = GenericMLP(
                input_dim=token_dim,
                hidden_dims=[token_dim],
                output_dim=token_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        self.if_mix_precision = if_mix_precision

        self.use_multi_layers = use_multi_layers
        self.depth_thres = depth_thres
        self.visualize_pred_pointcloud = visualize_pred_pointcloud
        self.pred_pointcloud_path = Path(pred_pointcloud_path)
        self.query_3d_nms_iou_thr = query_3d_nms_iou_thr
        self.query_min_points = query_min_points
        self.keyframe_count = keyframe_count


    @torch.no_grad()
    def extract_feat(self, batch_inputs_dict: dict):

        if self.vggt_encoder.training:
            for param in self.vggt_encoder.parameters():
                param.requires_grad = False

            self.vggt_encoder.eval()

        with torch.no_grad():
            with autocast(device):
                img = batch_inputs_dict['imgs'].float().div(255.0)
                aggregated_tokens_list, ps_idx = self.vggt_encoder.aggregator(img)
                return aggregated_tokens_list, ps_idx, img

    @torch.no_grad()
    def _save_pred_pointclouds(self, point_clouds, batch_data_samples):
        if not self.visualize_pred_pointcloud:
            return
        if (torch.distributed.is_available() and torch.distributed.is_initialized()
                and torch.distributed.get_rank() != 0):
            return

        self.pred_pointcloud_path.mkdir(parents=True, exist_ok=True)
        for point_cloud, data_sample in zip(point_clouds, batch_data_samples):
            image_path = data_sample.metainfo['img_path'][0]
            scene_name = Path(image_path).parent.name
            colors = torch.full(
                (len(point_cloud), 3), 255, dtype=point_cloud.dtype,
                device=point_cloud.device)
            points_with_color = torch.cat([point_cloud, colors], dim=-1)
            write_ply_rgb(
                points_with_color.detach().float().cpu().numpy(),
                str(self.pred_pointcloud_path / f'{scene_name}_omega_pred_points.ply'))

    @staticmethod
    def _original_image_shape(data_sample, view_index):
        image_shapes = data_sample.metainfo['ori_shape']
        if isinstance(image_shapes, torch.Tensor):
            image_shapes = image_shapes.tolist()
        if len(image_shapes) > 0 and isinstance(image_shapes[0], Number):
            return image_shapes
        return image_shapes[view_index]

    @staticmethod
    def _aligned_3d_nms(boxes, scores, iou_thr):
        order = scores.argsort(descending=True)
        keep = []
        while len(order) > 0:
            current = order[0]
            keep.append(current)
            if len(order) == 1:
                break
            remaining = order[1:]
            inter_min = torch.maximum(boxes[current, :3], boxes[remaining, :3])
            inter_max = torch.minimum(boxes[current, 3:], boxes[remaining, 3:])
            inter_size = (inter_max - inter_min).clamp_min(0)
            inter_volume = inter_size.prod(dim=-1)
            current_volume = (boxes[current, 3:] - boxes[current, :3]).prod()
            remaining_volume = (boxes[remaining, 3:] - boxes[remaining, :3]).prod(dim=-1)
            iou = inter_volume / (current_volume + remaining_volume - inter_volume).clamp_min(1e-6)
            order = remaining[iou <= iou_thr]
        return torch.stack(keep)

    @staticmethod
    @torch.no_grad()
    def _farthest_point_fill(points, existing_centers, num_samples):
        if num_samples <= 0 or len(points) == 0:
            return points.new_empty((0, 3))

        num_samples = min(num_samples, len(points))
        selected = torch.zeros(
            len(points), dtype=torch.bool, device=points.device)
        if len(existing_centers) > 0:
            min_distances = (
                (points[:, None, :] - existing_centers[None, :, :])
                .square().sum(dim=-1).min(dim=1).values)
        else:
            min_distances = points.square().sum(dim=-1)

        sampled_indices = []
        for _ in range(num_samples):
            current = min_distances.argmax()
            sampled_indices.append(current)
            selected[current] = True
            distances = (points - points[current]).square().sum(dim=-1)
            min_distances = torch.minimum(min_distances, distances)
            min_distances[selected] = -1
        return points[torch.stack(sampled_indices)]

    @torch.no_grad()
    def _select_keyframe_indices(self, aggregated_tokens_list, ps_idx, images):
        num_views = images.shape[1]
        if self.keyframe_count <= 0 or self.keyframe_count >= num_views:
            return None

        pose_enc = self.vggt_encoder.camera_head(
            aggregated_tokens_list, patch_token_start=ps_idx)
        extrinsics, _ = encoding_to_camera(pose_enc, images.shape[-2:])
        rotation = extrinsics[..., :3, :3].float()
        translation = extrinsics[..., :3, 3].float()
        camera_centers = -torch.matmul(
            rotation.transpose(-1, -2), translation.unsqueeze(-1)).squeeze(-1)
        keyframe_count = min(self.keyframe_count, num_views)
        batch_indices = []
        for centers in camera_centers:
            if not torch.isfinite(centers).all():
                indices = torch.linspace(
                    0, num_views - 1, keyframe_count,
                    device=centers.device).round().long()
                batch_indices.append(indices)
                continue

            selected = torch.zeros(num_views, dtype=torch.bool,
                                   device=centers.device)
            min_distances = torch.full(
                (num_views,), float('inf'), device=centers.device)
            current = 0
            indices = []
            for _ in range(keyframe_count):
                indices.append(current)
                selected[current] = True
                distances = (centers - centers[current]).square().sum(dim=-1)
                min_distances = torch.minimum(min_distances, distances)
                min_distances[selected] = -1
                current = int(min_distances.argmax().item())
            batch_indices.append(torch.tensor(
                sorted(indices), device=centers.device, dtype=torch.long))
        return batch_indices

    def _build_2d_box_queries(self, aggregated_tokens_list, ps_idx, images,
                              batch_inputs_dict, batch_data_samples):
        if 'view_2d_instances' not in batch_inputs_dict:
            raise RuntimeError('2D instances are required to build 3D box queries.')

        pose_enc = self.vggt_encoder.camera_head(
            aggregated_tokens_list, patch_token_start=ps_idx)
        extrinsic, intrinsic = encoding_to_camera(pose_enc, images.shape[-2:])
        depth_map, _ = self.vggt_encoder.dense_head(
            aggregated_tokens_list, images, patch_token_start=ps_idx)
        depth_map = depth_map.squeeze(-1)
        point_maps = unproject_depth_map_to_point_map_torch(
            depth_map, extrinsic, intrinsic)

        batch_size, _, height, width, _ = point_maps.shape
        norm_scale = torch.stack(batch_inputs_dict['avg_distance'], dim=0)
        point_maps = point_maps * norm_scale.to(point_maps).view(
            batch_size, 1, 1, 1, 1)
        batch_boxes = []
        batch_visual_points = []
        for batch_index, (data_sample, view_instances) in enumerate(
                zip(batch_data_samples, batch_inputs_dict['view_2d_instances'])):
            proposals = []
            proposal_scores = []
            visual_points = []
            for view_index, instances in enumerate(view_instances):
                original_shape = self._original_image_shape(
                    data_sample, view_index)
                original_height, original_width = original_shape[:2]
                boxes_2d = instances.bboxes.to(point_maps.device)
                scores_2d = instances.scores.to(point_maps.device)
                for box_2d, score in zip(boxes_2d, scores_2d):
                    x1 = max(int(torch.floor(box_2d[0] * width / original_width).item()), 0)
                    y1 = max(int(torch.floor(box_2d[1] * height / original_height).item()), 0)
                    x2 = min(int(torch.ceil(box_2d[2] * width / original_width).item()), width)
                    y2 = min(int(torch.ceil(box_2d[3] * height / original_height).item()), height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop_depth = depth_map[batch_index, view_index, y1:y2, x1:x2]
                    crop_points = point_maps[batch_index, view_index, y1:y2, x1:x2]
                    valid = torch.isfinite(crop_points).all(dim=-1)
                    valid &= crop_depth > 1e-5
                    valid &= crop_depth <= self.depth_thres
                    crop_points = crop_points[valid]
                    if len(crop_points) < self.query_min_points:
                        continue
                    if self.visualize_pred_pointcloud:
                        visual_points.append(crop_points)
                    box_min = crop_points.min(dim=0).values
                    box_max = crop_points.max(dim=0).values
                    if ((box_max - box_min) <= 1e-4).any():
                        continue
                    proposals.append(torch.cat([box_min, box_max]))
                    proposal_scores.append(score)

            if proposals:
                proposals = torch.stack(proposals)
                proposal_scores = torch.stack(proposal_scores)
                keep = self._aligned_3d_nms(
                    proposals, proposal_scores, self.query_3d_nms_iou_thr)
                proposals = proposals[keep[:self.num_queries]]
            else:
                proposals = point_maps.new_empty((0, 6))

            if len(proposals) < self.num_queries:
                keyframe_indices = batch_inputs_dict.get('keyframe_indices')
                if keyframe_indices is None:
                    view_indices = torch.arange(
                        point_maps.shape[1], device=point_maps.device)
                else:
                    view_indices = keyframe_indices[batch_index].to(
                        device=point_maps.device)
                sampled_points = point_maps[
                    batch_index, view_indices, ::16, ::16].reshape(-1, 3)
                sampled_depth = depth_map[
                    batch_index, view_indices, ::16, ::16].reshape(-1)
                valid_points = torch.isfinite(sampled_points).all(dim=-1)
                valid_points &= sampled_depth > 1e-5
                valid_points &= sampled_depth <= self.depth_thres
                sampled_points = sampled_points[valid_points]
                if len(sampled_points) == 0:
                    sampled_points = point_maps[
                        batch_index, view_indices].reshape(-1, 3)
                    full_depth = depth_map[
                        batch_index, view_indices].reshape(-1)
                    valid_points = torch.isfinite(sampled_points).all(dim=-1)
                    valid_points &= full_depth > 1e-5
                    valid_points &= full_depth <= self.depth_thres
                    sampled_points = sampled_points[valid_points]
                filler_points = self._farthest_point_fill(
                    sampled_points,
                    (proposals[:, :3] + proposals[:, 3:]) * 0.5,
                    self.num_queries - len(proposals))
                filler_boxes = torch.cat([filler_points, filler_points], dim=-1)
                proposals = torch.cat([proposals, filler_boxes])
                if len(proposals) < self.num_queries:
                    if len(proposals) == 0:
                        proposals = point_maps.new_zeros((1, 6))
                    proposals = torch.cat([
                        proposals,
                        proposals[-1:].expand(self.num_queries - len(proposals), -1)
                    ])
            batch_boxes.append(proposals)
            if visual_points:
                batch_visual_points.append(torch.cat(visual_points)[:100000])
            else:
                batch_visual_points.append(point_maps.new_zeros((0, 3)))

        query_boxes = torch.stack(batch_boxes)
        batch_inputs_dict['query_3d_boxes'] = query_boxes
        # self._save_pred_pointclouds(batch_visual_points, batch_data_samples)
        return (query_boxes[..., :3] + query_boxes[..., 3:]) * 0.5

    def _encode_query_centers(self, query_xyz):
        pos_embed = self.pos_embedding(query_xyz, input_range=None)
        return self.query_projection(pos_embed)


    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict,
                         images, batch_data_samples):

        if self.use_multi_layers:
            x = []
            for tokens in vggt_token_list:
                if tokens is None:
                    continue
                idx_layer = len(x)
                tokens_permute = tokens.permute(0, 3, 1, 2).contiguous()  
                patch_tokens = tokens_permute[:, :, :, ps_idx:]
                # patch_tokens_list.append(patch_tokens)
                if idx_layer == 0:
                    patch_tokens_projected = self.proj_feat_dim0(patch_tokens)
                elif idx_layer == 1:
                    patch_tokens_projected = self.proj_feat_dim1(patch_tokens)
                elif idx_layer == 2:
                    patch_tokens_projected = self.proj_feat_dim2(patch_tokens)
                elif idx_layer == 3:
                    patch_tokens_projected = self.proj_feat_dim3(patch_tokens)
                elif idx_layer == 4:
                    patch_tokens_projected = self.proj_feat_dim4(patch_tokens)
                # if not self.if_use_pred_pc_query:
                del patch_tokens

                batch_size, feat_dim, im_num, token_num = patch_tokens_projected.shape
                patch_tokens_projected = patch_tokens_projected.reshape(batch_size, feat_dim, -1)
                patch_tokens_projected = patch_tokens_projected.permute(2, 0, 1).contiguous() 
                x.append(patch_tokens_projected)

            if not self.if_use_pred_pc_query:
                del vggt_token_list
            
            
        else:
            tokens_last_layer = vggt_token_list[-1]
            patch_tokens_last_layer = tokens_last_layer[:, :, ps_idx:, :]  
            x = patch_tokens_last_layer.permute(0, 3, 1, 2).contiguous()
            x = self.proj_feat_dim(x)
            batch_size, feat_dim, im_num, token_num = x.shape
            x = x.reshape(batch_size, feat_dim, -1)
            x = x.permute(2, 0, 1).contiguous()

        if self.if_use_gt_query:
            query_xyz = torch.stack([
                data_sample.gt_instances_3d.bboxes_3d.tensor[:, :3]
                for data_sample in batch_data_samples
            ])
            query_embed = self._encode_query_centers(query_xyz)
            query_embed = query_embed.permute(2, 0, 1) # query_embed: [256, 4, 1024]
            tgt = torch.zeros((self.num_queries, batch_size, feat_dim), device=query_xyz.device)
            box_features = self.decoder(tgt, x, query_pos=query_embed, pos=None)[0]
            batch_inputs_dict['query_xyz'] = query_xyz
        elif self.if_use_pred_pc_query:
            query_xyz = self._build_2d_box_queries(
                vggt_token_list, ps_idx, images, batch_inputs_dict,
                batch_data_samples)
            query_embed = self._encode_query_centers(query_xyz)
            query_embed = query_embed.permute(2, 0, 1) # query_embed: [256, 4, 1024]
            tgt = torch.zeros((query_xyz.shape[1], batch_size, feat_dim), device=query_xyz.device)
            ######### idea 2 ############
            if self.if_task_query:
                expanded_task_query = self.task_query.unsqueeze(1).expand(-1, batch_size, -1) 
                tgt = torch.cat([tgt, expanded_task_query], dim=0)  # [num_queries+1, bs, feat_dim]
            ######### idea 2 ############

            box_features = self.decoder(tgt, x, query_pos=query_embed, pos=None, if_task_query=self.if_task_query)[0]
            batch_inputs_dict['query_xyz'] = query_xyz
        else:
            tgt = self.queries.unsqueeze(1).expand(-1, batch_size, -1) # [num_queries, batch_size, token_dim]
            box_features = self.decoder(tgt, x, query_pos=None, pos=None)[0]

        return box_features

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:

        vggt_token_list, ps_idx, img = self.extract_feat(batch_inputs_dict)
        self._add_view_2d_instances(
            batch_inputs_dict, batch_data_samples, vggt_token_list, ps_idx, img)

        if self.if_mix_precision:
            with autocast(device):
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)
        else: 
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)

        losses = self.bbox_head.loss(box_features, batch_data_samples, batch_inputs_dict, **kwargs) 
        return losses




    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        vggt_token_list, ps_idx, img = self.extract_feat(batch_inputs_dict)
        self._add_view_2d_instances(
            batch_inputs_dict, batch_data_samples, vggt_token_list, ps_idx, img)

        if self.if_mix_precision:
            with autocast(device):
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)
        else:
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]

        results_list = self.bbox_head.predict(box_features, batch_data_samples, batch_inputs_dict, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions


    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        vggt_token_list, ps_idx, img = self.extract_feat(batch_inputs_dict)
        self._add_view_2d_instances(
            batch_inputs_dict, batch_data_samples, vggt_token_list, ps_idx, img)

        if self.if_mix_precision:
            with autocast(device):
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)
        else:
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]

        results = self.bbox_head.forward(box_features, batch_inputs_dict)
        return results

    def _add_view_2d_instances(self, batch_inputs_dict, batch_data_samples,
                               aggregated_tokens_list, ps_idx, images):
        if self.two_d_detector is None:
            if self.if_use_pred_pc_query:
                raise RuntimeError(
                    'two_d_detector must be enabled for 2D-box 3D queries.')
            return
        keyframe_indices = self._select_keyframe_indices(
            aggregated_tokens_list, ps_idx, images)
        batch_inputs_dict['keyframe_indices'] = keyframe_indices
        batch_inputs_dict['view_2d_instances'] = self.two_d_detector(
            batch_data_samples, view_indices=keyframe_indices)

def build_decoder(args, if_multilevel=False):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )

    if if_multilevel:
         decoder = TransformerDecoder_Multilevel(
            decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
        )       
    else:
        decoder = TransformerDecoder(
            decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
        )
    return decoder
