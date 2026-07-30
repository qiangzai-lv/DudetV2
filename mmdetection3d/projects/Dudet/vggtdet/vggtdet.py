import sys
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from projects.Dudet.detr3_models.helpers import GenericMLP
from projects.Dudet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from projects.Dudet.detr3_models.utils.votenet_pc_util import write_ply_rgb
from projects.Dudet.vggtdet.device import autocast, get_device
from projects.Dudet.vggtdet.geometry_attention import GeometryAwareDeformableDecoder

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
            deformable_num_points=4,
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

        self.geometry_decoder = GeometryAwareDeformableDecoder(
            embed_dims=token_dim,
            num_layers=decoder_cfg['dec_nlayers'],
            num_heads=decoder_cfg['dec_nhead'],
            feedforward_channels=decoder_cfg['dec_ffn_dim'],
            num_feature_levels=4 if use_multi_layers else 1,
            num_points=deformable_num_points,
            dropout=decoder_cfg['dec_dropout'])

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
        self.geometry_queries = nn.Parameter(torch.empty(num_queries, token_dim))
        nn.init.xavier_normal_(self.geometry_queries)
        self.if_learnable_query = if_learnable_query

        if if_learnable_query:
            self.queries = nn.Parameter(torch.Tensor(num_queries, token_dim))
            nn.init.xavier_normal_(self.queries)

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
    def _build_pred_pc_fps_queries(self, aggregated_tokens_list, ps_idx, images,
                                   batch_inputs_dict, batch_data_samples):
        pose_enc = self.vggt_encoder.camera_head(
            aggregated_tokens_list, patch_token_start=ps_idx)
        extrinsic, intrinsic = encoding_to_camera(pose_enc, images.shape[-2:])
        depth_map, _ = self.vggt_encoder.dense_head(
            aggregated_tokens_list, images, patch_token_start=ps_idx)
        depth_map = depth_map.squeeze(-1)
        point_maps = unproject_depth_map_to_point_map_torch(
            depth_map, extrinsic, intrinsic)
        batch_size = point_maps.shape[0]
        norm_scale = torch.stack(batch_inputs_dict['avg_distance'], dim=0)
        point_maps = point_maps * norm_scale.to(point_maps).view(
            batch_size, 1, 1, 1, 1)

        batch_queries = []
        batch_reconstructed_points = []
        for batch_index in range(batch_size):
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
            batch_queries.append(
                self._farthest_point_sample(reconstructed_points, self.num_queries))
            batch_reconstructed_points.append(reconstructed_points)

        query_points = torch.stack(batch_queries)
        self._save_query_visualizations(
            batch_reconstructed_points, query_points, batch_data_samples)
        return query_points, extrinsic, intrinsic, norm_scale

    def _encode_query_centers(self, query_xyz):
        pos_embed = self.pos_embedding(query_xyz, input_range=None)
        return self.query_projection(pos_embed)

    def _build_patch_feature_maps(self, vggt_token_list, ps_idx, image_shape):
        patch_size = self.vggt_encoder.aggregator.patch_size
        patch_height = image_shape[0] // patch_size
        patch_width = image_shape[1] // patch_size
        feature_maps = []
        for tokens in vggt_token_list:
            if tokens is None:
                continue
            level = len(feature_maps)
            patch_tokens = tokens[:, :, ps_idx:, :].permute(0, 3, 1, 2)
            projector = getattr(self, f'proj_feat_dim{level}', None)
            if projector is None:
                projector = self.proj_feat_dim
            projected_tokens = projector(patch_tokens)
            batch_size, channels, num_views, num_tokens = projected_tokens.shape
            if num_tokens != patch_height * patch_width:
                raise ValueError('VGGT patch tokens do not match the input image grid')
            feature_maps.append(projected_tokens.permute(0, 2, 1, 3).reshape(
                batch_size, num_views, channels, patch_height, patch_width))
        return feature_maps

    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict,
                         images, batch_data_samples):

        query_xyz, extrinsics, intrinsics, coordinate_scale = self._build_pred_pc_fps_queries(
            vggt_token_list, ps_idx, images, batch_inputs_dict,
            batch_data_samples)
        feature_maps = self._build_patch_feature_maps(
            vggt_token_list, ps_idx, images.shape[-2:])
        query = self.geometry_queries.unsqueeze(0).expand(
            query_xyz.shape[0], -1, -1).to(dtype=feature_maps[0].dtype)
        batch_inputs_dict['query_xyz'] = query_xyz
        return self.geometry_decoder(
            query, feature_maps, query_xyz, extrinsics, intrinsics,
            coordinate_scale,
            images.shape[-2:], self.pos_embedding, self.query_projection,
            self.bbox_head.center_heads)


    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:

        vggt_token_list, ps_idx, img = self.extract_feat(batch_inputs_dict)

        if self.if_mix_precision:
            with autocast(device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    batch_data_samples)
        else: 
            box_features, refined_query_xyz = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)

        losses = self.bbox_head.loss(
            box_features, batch_data_samples, batch_inputs_dict,
            refined_query_xyz=refined_query_xyz, **kwargs)
        return losses




    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        vggt_token_list, ps_idx, img = self.extract_feat(batch_inputs_dict)

        if self.if_mix_precision:
            with autocast(device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    batch_data_samples)
        else:
            box_features, refined_query_xyz = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]
            refined_query_xyz = [refined_query_xyz[-1]] if refined_query_xyz else None
            layer_ids = [self.bbox_head.n_levels - 1]
        else:
            layer_ids = None

        results_list = self.bbox_head.predict(
            box_features, batch_data_samples, batch_inputs_dict,
            refined_query_xyz=refined_query_xyz, layer_ids=layer_ids,
            **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions


    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        vggt_token_list, ps_idx, img = self.extract_feat(batch_inputs_dict)

        if self.if_mix_precision:
            with autocast(device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    batch_data_samples)
        else:
            box_features, refined_query_xyz = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, batch_data_samples)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]
            refined_query_xyz = [refined_query_xyz[-1]] if refined_query_xyz else None
            layer_ids = [self.bbox_head.n_levels - 1]
        else:
            layer_ids = None

        results = self.bbox_head.forward(
            box_features, batch_inputs_dict, batch_data_samples,
            refined_query_xyz=refined_query_xyz, layer_ids=layer_ids)
        return results
