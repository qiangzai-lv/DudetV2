import sys
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from projects.Dudet.detr3_models.helpers import GenericMLP
from projects.Dudet.detr3_models.utils.votenet_pc_util import write_ply_rgb
from projects.Dudet.vggtdet.grounding_dino import GroundingDINO2DDetector
from projects.Dudet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from projects.Dudet.detr3_models.transformer import (TransformerDecoder, TransformerDecoder_Multilevel,
                                                     TransformerDecoderLayer)
_VGGT_OMEGA_ROOT = Path(__file__).resolve().parents[3] / 'vggt-omega'
if str(_VGGT_OMEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_OMEGA_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
vggt_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


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

@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Select spatially distributed query points without a custom CUDA extension."""
    batch_size, num_points, _ = xyz.shape
    if num_samples > num_points:
        raise ValueError("num_samples cannot exceed the point count")

    indices = torch.empty((batch_size, num_samples), dtype=torch.long, device=xyz.device)
    min_distances = xyz.new_full((batch_size, num_points), float("inf"))
    batch_indices = torch.arange(batch_size, device=xyz.device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)

    for sample_idx in range(num_samples):
        indices[:, sample_idx] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        squared_distances = (xyz - centroid).square().sum(dim=-1)
        min_distances = torch.minimum(min_distances, squared_distances)
        farthest = min_distances.max(dim=1).indices

    return indices


class LearnableQueries(nn.Module):
    def __init__(self, num_queries, channel):
        super().__init__()
        self.queries = nn.Parameter(torch.Tensor(num_queries, channel))
        nn.init.xavier_normal_(self.queries)
    
    def forward(self, batch_size):
        return self.queries.unsqueeze(0).expand(batch_size, -1, -1)
    

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
            if_use_atten_sample=False,
            atten_sample_ratio=10,
            depth_thres=1000,
            if_use_atten_fps=False,
            lambda_dist=1.0,
            if_task_query=False,
            if_add_noises=False,
            noise_level=None,
            vggt_omega_checkpoint='/mnt/workspace/pretrain/VGGT-Omega/vggt_omega_1b_512.pt',
            visualize_pred_pointcloud=False,
            pred_pointcloud_path='vis_dir/pred_points'
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
        self.if_use_atten_sample = if_use_atten_sample
        self.atten_sample_ratio = atten_sample_ratio
        self.depth_thres = depth_thres
        self.if_use_atten_fps = if_use_atten_fps
        self.lambda_dist = lambda_dist
        self.if_add_noises = if_add_noises
        self.noise_level = noise_level
        self.visualize_pred_pointcloud = visualize_pred_pointcloud
        self.pred_pointcloud_path = Path(pred_pointcloud_path)


    @torch.no_grad()
    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList, mode):

        if self.vggt_encoder.training:
            for param in self.vggt_encoder.parameters():
                param.requires_grad = False

            self.vggt_encoder.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=vggt_dtype):
                if self.if_use_atten_sample or self.if_use_atten_fps:
                    raise ValueError(
                        'VGGT-Omega does not provide patch attention scores.')
                img = batch_inputs_dict['imgs'].float().div(255.0)
                aggregated_tokens_list, ps_idx = self.vggt_encoder.aggregator(img)
                return aggregated_tokens_list, ps_idx, img, None



    @torch.no_grad()
    def batch_random_sample(self, points, k=100000, depth_mask=None, weights=None):
        B, N, _ = points.shape
        device = points.device
        
        rand_values = torch.rand(B, N, device=device)
        if depth_mask is not None:
            rand_values[depth_mask] = 0

        perm = torch.argsort(rand_values, dim=-1, descending=True)
        
        indices = perm[:, :k]
        
        batch_indices = torch.arange(B, device=device)[:, None]

        if weights is not None:
            return points[batch_indices, indices], weights[batch_indices, indices] 
        else:
            return points[batch_indices, indices]

    @torch.no_grad()
    def pred_pc_from_vggt(self, aggregated_tokens_list_ori, ps_idx, images,
                          batch_inputs_dict, images_patch_attn,
                          batch_data_samples):

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=vggt_dtype): 
                aggregated_tokens_list = [
                    tokens.contiguous() if tokens is not None else None
                    for tokens in aggregated_tokens_list_ori]

            with torch.cuda.amp.autocast(enabled=False):

                pose_enc = self.vggt_encoder.camera_head(
                    aggregated_tokens_list, patch_token_start=ps_idx)
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extrinsic, intrinsic = encoding_to_camera(
                    pose_enc, images.shape[-2:])

                
                depth_map, depth_conf = self.vggt_encoder.dense_head(
                    aggregated_tokens_list, images, patch_token_start=ps_idx)
                del aggregated_tokens_list

                assert depth_map.shape[-1] == 1
                depth_map = depth_map.squeeze(-1)
                
                if self.if_use_atten_sample:
                    images_patch_attn = images_patch_attn.float() 

                    point_map_by_unprojection_tensor = unproject_depth_map_to_point_map_torch(depth_map, extrinsic, intrinsic)
                    point_map_by_unprojection_tensor = point_map_by_unprojection_tensor.reshape(point_map_by_unprojection_tensor.shape[0], point_map_by_unprojection_tensor.shape[1], -1 ,point_map_by_unprojection_tensor.shape[-1]) # shape:(bs, view_num,  h * w, 3)

                    bs, num_frame, h, w  = depth_map.shape

                    depth_mask = depth_map > self.depth_thres  # shape [10, 40, 336, 448]

                    patch_size = self.vggt_encoder.aggregator.patch_size

                    attn_reshape = images_patch_attn.view(bs, num_frame, h//patch_size, w//patch_size)


                    attn_img_up = F.interpolate(attn_reshape,
                                size=(h, w),  # (H, W)
                                mode='bicubic',
                                align_corners=False)

                    attn_img_up = attn_img_up.view(bs*num_frame, -1) # (bs*num_frame, h*w)

                    min_val =  torch.min(attn_img_up, -1, keepdim=True).values # (bs*num_frame, 1)
                    max_val =  torch.max(attn_img_up, -1, keepdim=True).values # (bs*num_frame, 1)

                    denominator = max_val - min_val
                    norm_attn_img_up = torch.where( # (bs*num_frame, h*w)
                        denominator != 0,
                        (attn_img_up - min_val) / denominator,
                        torch.tensor(1.0, device=attn_img_up.device)
                    )
                    attn_depth_mask = depth_mask.view(bs*num_frame, -1)
                    norm_attn_img_up[attn_depth_mask] = 0.0 # (bs*num_frame, h*w)
                    prob_dist_pre = norm_attn_img_up
                    num_point = prob_dist_pre.shape[-1] # (bs*num_frame, h*w)
                    prob_dist = prob_dist_pre / prob_dist_pre.sum(dim=-1, keepdim=True)
                    num_samples = int(num_point / self.atten_sample_ratio)
                    topk_values, sampled_indices = torch.topk(prob_dist, num_samples, dim=1)


                    sampled_indices = sampled_indices.view(bs, num_frame, num_samples)
                    expanded_indices = sampled_indices.unsqueeze(-1).expand(-1, -1, -1, 3)

                    del norm_attn_img_up, attn_img_up, attn_reshape, prob_dist_pre, prob_dist
                    sampled_point_map_by_unprojection_tensor = torch.gather(point_map_by_unprojection_tensor, dim=2, index=expanded_indices)
                    sampled_point_map_by_unprojection_tensor = sampled_point_map_by_unprojection_tensor.reshape(sampled_point_map_by_unprojection_tensor.shape[0], -1, sampled_point_map_by_unprojection_tensor.shape[-1])
                    sampled_point_map_by_unprojection_tensor = self.batch_random_sample(sampled_point_map_by_unprojection_tensor, 100000)

                    del extrinsic, intrinsic, depth_map, depth_conf, pose_enc 

                elif self.if_use_atten_fps:
                    images_patch_attn = images_patch_attn.float() 

                    point_map_by_unprojection_tensor = unproject_depth_map_to_point_map_torch(depth_map, extrinsic, intrinsic)
                    point_map_by_unprojection_tensor = point_map_by_unprojection_tensor.reshape(point_map_by_unprojection_tensor.shape[0], -1 ,point_map_by_unprojection_tensor.shape[-1]) # shape:(bs, view_num,  h * w, 3)

                    bs, num_frame, h, w  = depth_map.shape

                    # # use depth 
                    depth_mask = depth_map > self.depth_thres  # shape [10, 40, 336, 448]
                    patch_size = self.vggt_encoder.aggregator.patch_size

                    attn_reshape = images_patch_attn.view(bs, num_frame, h//patch_size, w//patch_size)


                    attn_img_up = F.interpolate(attn_reshape,
                                size=(h, w),  # (H, W)
                                mode='bicubic',
                                align_corners=False)

                    attn_img_up = attn_img_up.view(bs, -1) # (bs, h*w*num_frame)

                    min_val =  torch.min(attn_img_up, -1, keepdim=True).values # (bs*num_frame, 1)
                    max_val =  torch.max(attn_img_up, -1, keepdim=True).values # (bs*num_frame, 1)

                    denominator = max_val - min_val
                    norm_attn_img_up = torch.where( # (bs*num_frame, h*w)
                        denominator != 0,
                        (attn_img_up - min_val) / denominator,
                        torch.tensor(1.0, device=attn_img_up.device)
                    )
                    attn_depth_mask = depth_mask.view(bs, -1)
                    norm_attn_img_up[attn_depth_mask] = 0.0 # (bs*num_frame, h*w)
                    prob_dist_pre = norm_attn_img_up
                    num_point = prob_dist_pre.shape[-1] # (bs*num_frame, h*w)
                    prob_dist = prob_dist_pre / prob_dist_pre.sum(dim=-1, keepdim=True)
                    # num_samples = int(num_point / self.atten_sample_ratio)
                    
                    prob_dist = prob_dist.view(bs, -1)

                    del norm_attn_img_up, attn_img_up, attn_reshape, prob_dist_pre

                    sampled_point_map_by_unprojection_tensor, atten_weights = self.batch_random_sample(point_map_by_unprojection_tensor, 100000, weights=prob_dist)

                    del extrinsic, intrinsic, depth_map, depth_conf, pose_enc, prob_dist
                    
                else:
                    point_map_by_unprojection_tensor = unproject_depth_map_to_point_map_torch(depth_map, extrinsic, intrinsic)
                    point_map_by_unprojection_tensor = point_map_by_unprojection_tensor.reshape(point_map_by_unprojection_tensor.shape[0], -1, point_map_by_unprojection_tensor.shape[-1])
                    depth_mask = depth_map > self.depth_thres
                    depth_mask = depth_mask.reshape(point_map_by_unprojection_tensor.shape[0], -1)

                    del extrinsic, intrinsic, depth_map, depth_conf, pose_enc 

                    sampled_point_map_by_unprojection_tensor = self.batch_random_sample(point_map_by_unprojection_tensor, 100000, depth_mask)

                norm_scale = batch_inputs_dict['avg_distance']
                norm_scale_tensor = torch.stack(norm_scale, dim=0).unsqueeze(-1)

                del point_map_by_unprojection_tensor
                sampled_point_map_by_unprojection_tensor *= norm_scale_tensor
                self._save_pred_pointclouds(
                    sampled_point_map_by_unprojection_tensor,
                    batch_data_samples)

                if self.if_use_atten_fps:
                    return sampled_point_map_by_unprojection_tensor.detach(), atten_weights
                else:
                    return sampled_point_map_by_unprojection_tensor.detach(), None

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
                points_with_color.detach().cpu().numpy(),
                str(self.pred_pointcloud_path / f'{scene_name}_omega_pred_points.ply'))


    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict,
                         images, images_patch_attn, batch_data_samples):

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
            points_xyz = torch.stack(batch_inputs_dict['points'], dim=0)[:, :, :3].contiguous()
            query_xyz, query_embed = self.get_query_embeddings(points_xyz, point_cloud_dims=None) # query_xyz shape: [4, 256, 3], query_embed: [4, 1024, 256]
            query_embed = query_embed.permute(2, 0, 1) # query_embed: [256, 4, 1024]
            tgt = torch.zeros((self.num_queries, batch_size, feat_dim), device=query_xyz.device)
            box_features = self.decoder(tgt, x, query_pos=query_embed, pos=None)[0]
            batch_inputs_dict['query_xyz'] = query_xyz
        elif self.if_use_pred_pc_query:
            pred_pc, atten_weights = self.pred_pc_from_vggt(
                vggt_token_list, ps_idx, images, batch_inputs_dict,
                images_patch_attn, batch_data_samples)
            if self.if_add_noises:
                pred_pc = self.add_normalized_noise_to_point_cloud(pred_pc, self.noise_level)
            if self.if_use_atten_fps:
                query_xyz, query_embed = self.get_query_embeddings_atten_fps(pred_pc, point_cloud_dims=None, atten_weights=atten_weights)
            else:
                query_xyz, query_embed = self.get_query_embeddings(pred_pc, point_cloud_dims=None) # query_xyz shape: [4, 256, 3], query_embed: [4, 1024, 256]

            query_embed = query_embed.permute(2, 0, 1) # query_embed: [256, 4, 1024]
            tgt = torch.zeros((self.num_queries, batch_size, feat_dim), device=query_xyz.device)
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

    def add_normalized_noise_to_point_cloud(self, pred_pc, noise_level):

        assert len(pred_pc.shape) == 3 and pred_pc.shape[2] == 3, "the shape of pred_pc should be [1, N, 3]"
        assert 0.0 <= noise_level <= 1.0, "the noise_level must be in the range [0, 1]"
        
        max_coords = torch.max(pred_pc, dim=1, keepdim=True)[0]  # [1, 1, 3]
        min_coords = torch.min(pred_pc, dim=1, keepdim=True)[0]  # [1, 1, 3]
        range_coords = max_coords - min_coords                  # [1, 1, 3]
        
        actual_std = range_coords * noise_level
        
        noise = torch.randn_like(pred_pc) * actual_std
        
        noisy_pc = pred_pc + noise
        
        return noisy_pc

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:

        if self.two_d_detector is not None:
            batch_inputs_dict["view_2d_instances"] = self.two_d_detector(
                batch_data_samples)

        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with torch.cuda.amp.autocast(dtype=vggt_dtype):
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn, batch_data_samples)
        else: 
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn, batch_data_samples)

        losses = self.bbox_head.loss(box_features, batch_data_samples, batch_inputs_dict, **kwargs) 
        return losses




    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with torch.cuda.amp.autocast(dtype=vggt_dtype):
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn, batch_data_samples)
        else:
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn, batch_data_samples)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]

        results_list = self.bbox_head.predict(box_features, batch_data_samples, batch_inputs_dict, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions


    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with torch.cuda.amp.autocast(dtype=vggt_dtype):
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn, batch_data_samples)
        else:
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn, batch_data_samples)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]

        results = self.bbox_head.forward(box_features, batch_inputs_dict)
        return results

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = farthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def get_query_embeddings_atten_fps(self, encoder_xyz, point_cloud_dims, atten_weights=None):
        query_inds = self.attention_guided_prob_fps(encoder_xyz, atten_weights, self.num_queries, lambda_dist=self.lambda_dist)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed



    @torch.no_grad()
    def attention_guided_prob_fps(self,
        points: torch.Tensor,
        attention_weights: torch.Tensor,
        num_samples: int,
        lambda_dist: float = 0.1,
        chunk_size: int = 16384,
        use_amp: bool = True,
        verbose: bool = False
    ) -> torch.Tensor:

        assert points.dim() == 3, "the shape of points should be [B, N, 3]"
        assert attention_weights.shape == points.shape[:2], "the shape of attention_weights should be [B, N]"
        
        with torch.cuda.amp.autocast(dtype=vggt_dtype if use_amp else None): 
            B, N, _ = points.shape
            device = points.device
            batch_idx = torch.arange(B, device=device)[:, None]
            
            indices = torch.zeros((B, num_samples), dtype=torch.long, device=device)
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
            
            weights_min = attention_weights.min(1, keepdim=True).values
            weights_max = attention_weights.max(1, keepdim=True).values
            weights_norm = (attention_weights - weights_min) / (weights_max - weights_min + 1e-8)
            weights_norm = weights_norm.to(vggt_dtype)
            
            first_idx = torch.argmax(weights_norm, dim=1)
            indices[:, 0] = first_idx
            mask[batch_idx, first_idx.unsqueeze(1)] = False
            
            min_dists = torch.full((B, N), float('inf'), dtype=vggt_dtype, device=device)
            
            for k in range(1, num_samples):
                current_point = points.gather(1, indices[:, k-1].view(-1,1,1).expand(-1,-1,3))
                for i in range(0, N, chunk_size):
                    chunk = points[:, i:i+chunk_size]
                    dist_chunk = torch.norm(chunk - current_point, dim=-1)
                    min_dists[:, i:i+chunk_size] = torch.min(
                        min_dists[:, i:i+chunk_size], 
                        dist_chunk.to(vggt_dtype)
                    )
                
                dist_min = min_dists.min(1, keepdim=True).values
                dist_max = min_dists.max(1, keepdim=True).values
                dists_norm = (min_dists - dist_min) / (dist_max - dist_min + 1e-8)
                
                priority = weights_norm + lambda_dist * dists_norm
                priority[~mask] = -torch.inf
                
                next_idx = torch.argmax(priority, dim=1)
                indices[:, k] = next_idx
                mask[batch_idx, next_idx.unsqueeze(1)] = False
                
                if verbose and k % 10 == 0:
                    mem = torch.cuda.memory_allocated() / 1024**3
                    print(f"Step {k}: Mem {mem:.2f}GB | Min Dist {min_dists.min().item():.4f}")
        
        return indices

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
