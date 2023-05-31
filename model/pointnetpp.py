import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from model.modules.pointnetpp_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from thirdpart.chamfer_dist import ChamferDistanceL2


# Backbone for pre-training and semantic segmentation
class PointNetPP(nn.Module):
    def __init__(self, num_feats, model_n_out, config, D):
        super().__init__()
        self.in_channel = config.net.in_channel
        if self.in_channel == 3: feed_channel = 0
        else: feed_channel = self.in_channel
        self.num_classes = config.num_classes
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], feed_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=131, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.return_fea = config.net.return_fea
        self.conv_embed = torch.nn.Conv1d(128, model_n_out, 1)    

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2)
        B,C,N = xyz.shape
        if self.in_channel and self.in_channel > 3:
            l0_xyz = xyz[:, :3, :]
            l0_points = xyz
        else:
            l0_xyz = xyz
            l0_points = None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_xyz, l1_points)
        # FC layers
        tmp_feat = self.bn1(self.conv1(l0_points))
        feat = F.relu(tmp_feat)
        if self.return_fea:
            return feat
        x = self.drop1(feat)
        x = self.conv_embed(x)
        return x


# Backbone for object classfication
class PointNetPPEnc(nn.Module):
    def __init__(self, num_feats, model_n_out, config, D):
        super().__init__()
        self.in_channel = config.net.in_channel
        if self.in_channel == 3: feed_channel = 0
        else: feed_channel = self.in_channel
        self.num_classes = config.num_classes
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], feed_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.cls_fc1 = nn.Linear(1024, 512)
        self.bns1 = nn.BatchNorm1d(512)
        self.drops1 = nn.Dropout(0.5)
        self.cls_fc2 = nn.Linear(512, 256)
        self.bns2 = nn.BatchNorm1d(256)
        self.drops2 = nn.Dropout(0.5)
        self.cls_fc3 = nn.Linear(256, model_n_out)   

    def forward(self, xyz, layer=1):
        xyz = xyz.transpose(1, 2)
        B,C,N = xyz.shape
        if self.in_channel and self.in_channel > 3:
            #xyz = xyz[:, :3, :]
            #norm = xyz[:, 3:, :]
            l0_xyz = xyz[:, 3:, :]
            l0_points = xyz
        else:
            l0_xyz = xyz
            l0_points = None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        if layer == 0: return x
        x = self.drops1(F.relu(self.bns1(self.cls_fc1(x))))
        x = self.drops2(F.relu(self.bns2(self.cls_fc2(x))))
        x = self.cls_fc3(x)
        return x


# Backbone for part segmentation 
class PointNetPPEncDec(nn.Module):
    def __init__(self, num_feats, model_n_out, config, D):
        super().__init__()
        self.in_channel = 0
        if self.in_channel == 3: feed_channel = 0
        else: feed_channel = self.in_channel
        self.num_parts = config.num_parts
        self.num_classes = config.num_classes
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], feed_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=131, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.multi_shape_heads = nn.ModuleList()
        for i in range(self.num_classes):
            self.multi_shape_heads.append(
                nn.Sequential(nn.Conv1d(128, self.num_parts[i], 1)))        

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2)
        B,C,N = xyz.shape
        if self.in_channel and self.in_channel > 3:
            l0_xyz = xyz[:, 3:, :]
            l0_points = xyz
        else:
            l0_xyz = xyz
            l0_points = None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        #cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_xyz, l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)

        logits_all_shapes = []
        for i in range(self.num_classes):
            logits_all_shapes.append(self.multi_shape_heads[i](x))

        return logits_all_shapes

