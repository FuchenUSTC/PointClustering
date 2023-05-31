import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

from .modules.transformer import Mlp, Attention
from .modules.pointnetpp_utils import PointNetFeaturePropagation
from util.util import fps


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, _ = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerPartSegEncoder(TransformerEncoder):
    """ Transformer Encoder with hierarchical output
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(TransformerPartSegEncoder, self).__init__(embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate)

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Backbone for pre-training and semantic segmentation
class PointTransformerNet(nn.Module):
    def __init__(self, num_feats, model_n_out, config, D):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes

        self.cls_dim = config.net.model_n_out
        self.trans_dim = config.net.trans_dim
        self.depth = config.net.depth
        self.drop_path_rate = config.net.drop_path_rate
        self.num_heads = config.net.num_heads

        self.group_size = config.net.group_size
        self.num_group = config.net.num_group
        self.encoder_dims = config.net.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerPartSegEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])

        self.convs1 = nn.Conv1d(1024, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.drops1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 128, 1)
        self.bns2 = nn.BatchNorm1d(128)
        self.conv_embed = torch.nn.Conv1d(128, model_n_out, 1)    

        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        neighborhood, center = self.group_divider(xyz) # B N C
        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)
        x = group_input_tokens
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #B 1152 N

        f_level_0 = self.propagation_0(xyz.transpose(-1, -2), center.transpose(-1, -2), xyz.transpose(-1, -2), x)

        x = self.relu(self.bns1(self.convs1(f_level_0)))
        x = self.drops1(x)
        x = self.relu(self.bns2(self.convs2(x)))   
        x = self.conv_embed(x)

        return x


# Backbone for object classfication
class PointTransformerEncNet(nn.Module):
    def __init__(self, num_feats, model_n_out, config, D):
        super().__init__()
        self.config = config

        self.cls_dim = config.net.model_n_out
        self.trans_dim = config.net.trans_dim
        self.depth = config.net.depth
        self.drop_path_rate = config.net.drop_path_rate
        self.num_heads = config.net.num_heads

        self.group_size = config.net.group_size
        self.num_group = config.net.num_group
        self.encoder_dims = config.net.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        neighborhood, center = self.group_divider(xyz) # B N 3
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret


# Backbone for part segmentation 
class PointTransformerPartSegNet(nn.Module):
    def __init__(self, num_feats, model_n_out, config, D):
        super().__init__()
        self.config = config
        self.num_parts = config.num_parts
        self.num_classes = config.num_classes

        self.cls_dim = config.net.model_n_out
        self.trans_dim = config.net.trans_dim
        self.depth = config.net.depth
        self.drop_path_rate = config.net.drop_path_rate
        self.num_heads = config.net.num_heads

        self.group_size = config.net.group_size
        self.num_group = config.net.num_group
        self.encoder_dims = config.net.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerPartSegEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])

        self.convs1 = nn.Conv1d(1024, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.drops1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 128, 1)
        self.bns2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

        self.multi_shape_heads = nn.ModuleList()
        for i in range(self.num_classes):
            self.multi_shape_heads.append(
                nn.Sequential(nn.Conv1d(128, self.num_parts[i], 1)))    

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        neighborhood, center = self.group_divider(xyz) # B N C
        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)
        x = group_input_tokens
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #B 1152 N

        f_level_0 = self.propagation_0(xyz.transpose(-1, -2), center.transpose(-1, -2), xyz.transpose(-1, -2), x)

        x = self.relu(self.bns1(self.convs1(f_level_0)))
        x = self.drops1(x)
        x = self.relu(self.bns2(self.convs2(x)))   

        logits_all_shapes = []
        for i in range(self.num_classes):
            logits_all_shapes.append(self.multi_shape_heads[i](x))

        return logits_all_shapes
