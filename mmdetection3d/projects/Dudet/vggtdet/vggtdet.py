import sys
from numbers import Number
from pathlib import Path
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
            query_fps_stride=16,
            query_fps_max_points=100000,
            visualize_query_points=False,
            query_visualization_path='vis_dir/query_points',
            query_visualization_marker_size=0.05
            ):
        
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg) 

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
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
        self.query_fps_stride = query_fps_stride
        self.query_fps_max_points = query_fps_max_points
        self.visualize_query_points = visualize_query_points
        self.query_visualization_path = Path(query_visualization_path)
        self.query_visualization_marker_size = query_visualization_marker_size
        self._visualized_query_scenes = set()


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

    @staticmethod
    def _original_image_shape(data_sample, view_index):
        image_shapes = data_sample.metainfo['ori_shape']
        if isinstance(image_shapes, torch.Tensor):
            image_shapes = image_shapes.tolist()
        if len(image_shapes) > 0 and isinstance(image_shapes[0], Number):
            return image_shapes
        return image_shapes[view_index]

    @staticmethod
    def _project_gt_instances(data_sample, view_index, image_shape, device):
        gt_instances = data_sample.gt_instances_3d
        corners = gt_instances.bboxes_3d.corners.to(
            device=device, dtype=torch.float32)
        labels = gt_instances.labels_3d.to(device=device, dtype=torch.long)
        if len(corners) == 0:
            return None

        lidar2img = data_sample.metainfo['lidar2img']
        extrinsic = torch.as_tensor(
            lidar2img['extrinsic'][view_index], dtype=torch.float32,
            device=device)
        intrinsics = torch.as_tensor(
            lidar2img['intrinsic'], dtype=torch.float32, device=device)
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
        valid &= torch.isfinite(pixels).all(dim=-1)
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
            return None
        return torch.stack(bboxes), torch.stack(projected_labels)

    @staticmethod
    @torch.no_grad()
    def _farthest_point_sample(points, num_samples):
        if len(points) == 0:
            return points.new_zeros((num_samples, 3))

        sample_count = min(num_samples, len(points))
        indices = torch.empty(
            sample_count, dtype=torch.long, device=points.device)
        min_distances = torch.full(
            (len(points),), float('inf'), device=points.device)
        current = points.square().sum(dim=-1).argmax()
        for sample_index in range(sample_count):
            indices[sample_index] = current
            distances = (points - points[current]).square().sum(dim=-1)
            min_distances = torch.minimum(min_distances, distances)
            current = min_distances.argmax()
        sampled_points = points[indices]
        if sample_count < num_samples:
            sampled_points = torch.cat([
                sampled_points,
                sampled_points[-1:].expand(num_samples - sample_count, -1)
            ])
        return sampled_points

    @torch.no_grad()
    def _save_query_visualizations(self, reconstructed_points, query_points,
                                   batch_data_samples):
        if not self.visualize_query_points:
            return
        if (torch.distributed.is_available() and torch.distributed.is_initialized()
                and torch.distributed.get_rank() != 0):
            return

        self.query_visualization_path.mkdir(parents=True, exist_ok=True)
        for reconstructed, queries, data_sample in zip(
                reconstructed_points, query_points, batch_data_samples):
            scene_name = Path(data_sample.metainfo['img_path'][0]).parent.name
            if scene_name in self._visualized_query_scenes:
                continue
            self._visualized_query_scenes.add(scene_name)

            reconstruction_color = torch.tensor(
                [80, 170, 255], dtype=reconstructed.dtype,
                device=reconstructed.device).expand(len(reconstructed), -1)
            marker_offsets = torch.eye(
                3, dtype=queries.dtype, device=queries.device)
            marker_offsets = torch.cat([
                torch.zeros_like(marker_offsets[:1]), marker_offsets,
                -marker_offsets
            ]) * self.query_visualization_marker_size
            query_markers = (queries[:, None, :] + marker_offsets[None, :, :])
            query_markers = query_markers.reshape(-1, 3)
            query_color = torch.tensor(
                [255, 70, 70], dtype=query_markers.dtype,
                device=query_markers.device).expand(len(query_markers), -1)
            write_ply_rgb(
                torch.cat([
                    torch.cat([reconstructed, reconstruction_color], dim=-1),
                    torch.cat([query_markers, query_color], dim=-1)
                ], dim=0)
                .detach().float().cpu().numpy(),
                str(self.query_visualization_path /
                    f'{scene_name}_reconstruction_queries.ply'))

    @torch.no_grad()
    def _build_gt_fps_queries(self, aggregated_tokens_list, ps_idx, images,
                              batch_inputs_dict, batch_data_samples):
        pose_enc = self.vggt_encoder.camera_head(
            aggregated_tokens_list, patch_token_start=ps_idx)
        extrinsic, intrinsic = encoding_to_camera(pose_enc, images.shape[-2:])
        depth_map, _ = self.vggt_encoder.dense_head(
            aggregated_tokens_list, images, patch_token_start=ps_idx)
        depth_map = depth_map.squeeze(-1)
        point_maps = unproject_depth_map_to_point_map_torch(
            depth_map, extrinsic, intrinsic)
        batch_size, num_views, height, width, _ = point_maps.shape
        norm_scale = torch.stack(batch_inputs_dict['avg_distance'], dim=0)
        point_maps = point_maps * norm_scale.to(point_maps).view(
            batch_size, 1, 1, 1, 1)

        batch_queries = []
        batch_projected_gt = []
        batch_reconstructed_points = []
        for batch_index, data_sample in enumerate(batch_data_samples):
            candidate_points = []
            view_bboxes = []
            for view_index in range(num_views):
                original_shape = self._original_image_shape(
                    data_sample, view_index)
                projected_gt = self._project_gt_instances(
                    data_sample, view_index, original_shape, point_maps.device)
                if projected_gt is None:
                    view_bboxes.append(None)
                    continue
                boxes_2d, _ = projected_gt
                view_bboxes.append(boxes_2d)
                original_height, original_width = original_shape[:2]
                for box_2d in boxes_2d:
                    if (original_height <= 0 or original_width <= 0
                            or not torch.isfinite(box_2d).all()):
                        continue
                    x1 = max(int(torch.floor(box_2d[0] * width / original_width).item()), 0)
                    y1 = max(int(torch.floor(box_2d[1] * height / original_height).item()), 0)
                    x2 = min(int(torch.ceil(box_2d[2] * width / original_width).item()), width)
                    y2 = min(int(torch.ceil(box_2d[3] * height / original_height).item()), height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop_points = point_maps[
                        batch_index, view_index, y1:y2:self.query_fps_stride,
                        x1:x2:self.query_fps_stride]
                    crop_depth = depth_map[
                        batch_index, view_index, y1:y2:self.query_fps_stride,
                        x1:x2:self.query_fps_stride]
                    valid = torch.isfinite(crop_points).all(dim=-1)
                    valid &= crop_depth > 1e-5
                    valid &= crop_depth <= self.depth_thres
                    if valid.any():
                        candidate_points.append(crop_points[valid])

            if candidate_points:
                candidate_points = torch.cat(candidate_points)
            else:
                fallback_points = point_maps[
                    batch_index, :, ::self.query_fps_stride,
                    ::self.query_fps_stride].reshape(-1, 3)
                fallback_depth = depth_map[
                    batch_index, :, ::self.query_fps_stride,
                    ::self.query_fps_stride].reshape(-1)
                valid = torch.isfinite(fallback_points).all(dim=-1)
                valid &= fallback_depth > 1e-5
                valid &= fallback_depth <= self.depth_thres
                candidate_points = fallback_points[valid]
            reconstructed_points = point_maps[
                batch_index, :, ::self.query_fps_stride,
                ::self.query_fps_stride].reshape(-1, 3)
            reconstructed_depth = depth_map[
                batch_index, :, ::self.query_fps_stride,
                ::self.query_fps_stride].reshape(-1)
            valid = torch.isfinite(reconstructed_points).all(dim=-1)
            valid &= reconstructed_depth > 1e-5
            valid &= reconstructed_depth <= self.depth_thres
            reconstructed_points = reconstructed_points[valid]
            if len(reconstructed_points) > self.query_fps_max_points:
                step = (len(reconstructed_points) + self.query_fps_max_points - 1)
                step //= self.query_fps_max_points
                reconstructed_points = reconstructed_points[::step]
            if len(candidate_points) > self.query_fps_max_points:
                step = (len(candidate_points) + self.query_fps_max_points - 1)
                step //= self.query_fps_max_points
                candidate_points = candidate_points[::step]
            batch_queries.append(
                self._farthest_point_sample(candidate_points, self.num_queries))
            batch_projected_gt.append(view_bboxes)
            batch_reconstructed_points.append(reconstructed_points)

        batch_inputs_dict['view_2d_gt_bboxes'] = batch_projected_gt
        query_points = torch.stack(batch_queries)
        self._save_query_visualizations(
            batch_reconstructed_points, query_points, batch_data_samples)
        return query_points

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
            query_xyz = self._build_gt_fps_queries(
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

        if self.if_mix_precision:
            with autocast(device):
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)
        else:
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]

        results = self.bbox_head.forward(box_features, batch_inputs_dict)
        return results

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
