import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCrossAttention(nn.Module):
    """Full cross-attention over all view and feature-level patch tokens."""

    def __init__(self, embed_dims, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, query_pos, feature_maps, *args):
        _, _, channels = query.shape
        values = []
        for feature_map in feature_maps:
            _, _, feature_channels, _, _ = feature_map.shape
            if feature_channels != channels:
                raise ValueError("Feature-map channels do not match queries")
            values.append(feature_map.permute(0, 1, 3, 4, 2).reshape(feature_map.shape[0], -1, channels))
        value = torch.cat(values, dim=1)
        return self.attention(
            query=query + query_pos, key=value, value=value,
            need_weights=False)[0]


class GeometryAwareDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels,
                 num_feature_levels, num_points, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = StandardCrossAttention(embed_dims, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, query_pos, feature_maps):
        query_norm = self.norm1(query)
        self_attended = self.self_attn(
            query_norm + query_pos, query_norm + query_pos, query_norm,
            need_weights=False)[0]
        query = query + self.dropout(self_attended)
        cross_attended = self.cross_attn(
            self.norm2(query), query_pos, feature_maps)
        query = query + self.dropout(cross_attended)
        ffn = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(query)))))
        return query + self.dropout(ffn)


class GeometryAwareDeformableDecoder(nn.Module):
    def __init__(self, embed_dims, num_layers, num_heads, feedforward_channels,
                 num_feature_levels, num_points=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GeometryAwareDecoderLayer(
                embed_dims, num_heads, feedforward_channels,
                num_feature_levels, num_points, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, query, feature_maps, reference_points, extrinsics,
                intrinsics, coordinate_scale, image_shape,
                position_embedding, query_projection, reg_branches):
        if len(reg_branches) != len(self.layers):
            raise ValueError('Each decoder layer requires one regression branch')

        intermediate = []
        intermediate_references = []
        for layer_id, layer in enumerate(self.layers):
            query_xyz = reference_points
            query_pos = query_projection(
                position_embedding(query_xyz, input_range=None)).transpose(1, 2)
            query = layer(query, query_pos, feature_maps)
            output = self.norm(query)
            delta_xyz = reg_branches[layer_id](output.transpose(1, 2)).transpose(1, 2)
            reference_points = reference_points + delta_xyz
            intermediate.append(output.transpose(1, 2))
            intermediate_references.append(reference_points)
            reference_points = reference_points.detach()
        return intermediate, intermediate_references
