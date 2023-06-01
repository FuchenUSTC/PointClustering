# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2022/05/08
# desc: pre-train data loader


import random
import itertools
import math
import torch
import torch.utils.data
import torch.distributed as dist
import os
import copy
import numpy as np
import pandas as pd
import MinkowskiEngine as ME
import dataset.transforms as t
import util.distributed as du
import open3d as o3d
from random import shuffle
from scipy.linalg import expm, norm
from torchvision import transforms as torchtransform
from sklearn import cluster
from collections import Counter
from .data_util import *


# ----- Pre-training Task Dataset -------- #

class ScanNetMatchPairDataset(torch.utils.data.Dataset):
  def __init__(self,
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.transform = transform
    self.voxel_size = config.data.voxel_size
    self.matching_search_voxel_size = \
        config.data.voxel_size * config.trainer.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.trainer.min_scale
    self.max_scale = config.trainer.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.trainer.rotation_range
    self.randg = np.random.RandomState()
    self.logger = logger
    self.config = config
    
    if manual_seed:
      self.reset_seed()
    
    self.root_filelist = root = config.data.scannet_match_dir
    self.root = config.data.dataset_root_dir
    is_master = du.is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True
    if is_master: logger.info(f"Loading the subset {phase} from {root}")
    if phase == "train":
       fname_txt = os.path.join(self.root, self.root_filelist)
       with open(fname_txt) as f:
         content = f.readlines()
       fnames = [x.strip().split() for x in content]
       for fname in fnames:
         self.files.append([fname[0], fname[1]])
    else:
        raise NotImplementedError
    self.logger.info('train dataset has %d data pairs.'%(len(self.files)))
    
  def reset_seed(self, seed=0):
    self.logger.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)
 
  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]
    
    #dummy color
    color0 = np.ones((xyz0.shape[0], 3))
    color1 = np.ones((xyz1.shape[0], 3))

    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = apply_transform(xyz0, T0)
      xyz1 = apply_transform(xyz1, T1)
    else:
      trans = np.identity(4)

    # Voxelization
    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    # Get features
    npts0 = len(pcd0.colors)
    npts1 = len(pcd1.colors)

    feats_train0, feats_train1 = [], []

    feats_train0.append(np.ones((npts0, 3)))
    feats_train1.append(np.ones((npts1, 3)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class ScanNetMatchPairPointDataset(ScanNetMatchPairDataset):
  def __init__(self, 
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    super(ScanNetMatchPairPointDataset, self).__init__(phase, logger, transform, 
    random_rotation, random_scale, manual_seed, config)
    if self.config.data.sampler:
      param = {}
      param['n_points'] = self.config.data.num_sample
      self.sampler = torchtransform.Compose([t.RandomSamplePoints(parameters=param)])
  
  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]

    if self.config.data.sampler:
      xyz0 = self.sampler(xyz0)
      xyz1 = self.sampler(xyz1)

    xyz0, xyz1 = pair_pc_normalize(xyz0, xyz1)

    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = apply_transform(xyz0, T0)
      xyz1 = apply_transform(xyz1, T1)
    else:
      trans = np.identity(4)

    # Make point clouds
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    return (xyz0, xyz1, matches, trans)


class ScanNetRanSacPointDataset(ScanNetMatchPairDataset):
  def __init__(self, 
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    super(ScanNetRanSacPointDataset, self).__init__(phase, logger, transform, 
    random_rotation, random_scale, manual_seed, config)
    if self.config.data.sampler:
      param = {}
      param['n_points'] = self.config.data.num_sample
      self.sampler = torchtransform.Compose([t.RandomSamplePoints(parameters=param)]) 
    self.files = []
    is_master = du.is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True
    if is_master: logger.info(f"Loading the subset {phase} from {self.root}")
    if phase == "train":
       fname_txt = os.path.join(self.root, self.root_filelist)
       with open(fname_txt) as f:
         content = f.readlines()
       fnames = [x.strip().split() for x in content]
       for fname in fnames:
         if fname[0] not in self.files:
           self.files.append(fname[0])
         if fname[1] not in self.files:
           self.files.append(fname[1])
    else:
        raise NotImplementedError    
    self.logger.info('train dataset has %d deduplicated data.'%(len(self.files)))

  def ground_ransac(self, pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    obj_pcd = pcd.select_by_index(inliers, invert=True)
    all_idx = np.ones(len(pcd.points))
    all_idx[inliers] = 0
    obj_idx = np.argwhere(all_idx == 1)
    obj_idx = obj_idx.reshape(obj_idx.shape[0])
    return obj_pcd, inliers, obj_idx

  def dbscan(self, xyz, eps=0.08, min_num = 20):
      clustering = cluster.DBSCAN(eps=eps, min_samples=15).fit(xyz)
      labels = clustering.labels_
      # remove noise label
      num_dict = Counter(labels)
      for key in num_dict: 
        if num_dict[key] < min_num: 
          idx = np.argwhere(labels == key)
          labels[idx] = -1
      labels_denoise_idx = np.argwhere(labels != -1)
      labels_denoise_idx = labels_denoise_idx.reshape(labels_denoise_idx.shape[0])
      labels_denoise = labels[labels_denoise_idx]
      return labels_denoise, labels_denoise_idx

  def raw_dbscan(self, xyz, eps=0.09, min_samples=15):
      clustering = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
      labels = clustering.labels_
      return labels, np.arange(labels.shape[0])

  def get_matches(self, xyz_ref, psud_labels, psud_pc_idx, search_voxel_size):
    source_copy = copy.deepcopy(xyz_ref)
    target_copy = copy.deepcopy(xyz_ref)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
      [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
      for j in idx:
        if psud_labels[i] == psud_labels[j] and i != j:
          match_inds.append((psud_pc_idx[i], psud_pc_idx[j]))
    return match_inds
    
  def __getitem__(self, idx):
    file = os.path.join(self.root, self.files[idx])
    data = np.load(file)
    xyz = data["pcd"]
    xyz = pc_normalize(xyz)
    
    # Ransac to obtain pseudo label
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(xyz)
    #obj_pcd, grd_idx, obj_idx = self.ground_ransac(pcd)
    #psud_labels, psud_labels_id = self.dbscan(obj_pcd.points)
    #psud_pc_idx = np.array(obj_idx)[psud_labels_id]
    #assert len(psud_labels) == len(psud_pc_idx)
    #xyz_ref = xyz[psud_pc_idx].reshape(-1, 3)
    #pcd.points = o3d.utility.Vector3dVector(xyz_ref)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    psud_labels, psud_pc_idx = self.raw_dbscan(xyz)

    # get match label pair
    matches = self.get_matches(pcd, psud_labels, psud_pc_idx, self.matching_search_voxel_size)

    if self.random_rotation:
      T = sample_random_trans(xyz, self.randg, self.rotation_range)
      trans = np.linalg.inv(T)
      xyz = apply_transform(xyz, T)
    else:
      trans = np.identity(4)

    return (xyz, matches, trans)


# instance-level pseudo label
class ScanNetPseudoPointDataset(ScanNetRanSacPointDataset):
  def __init__(self, 
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.transform = transform
    self.voxel_size = config.data.voxel_size
    self.matching_search_voxel_size = \
        config.data.voxel_size * config.trainer.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.trainer.min_scale
    self.max_scale = config.trainer.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.trainer.rotation_range
    self.randg = np.random.RandomState()
    self.logger = logger
    self.config = config
    
    if manual_seed:
      self.reset_seed()
    
    self.root_filelist = config.data.scannet_match_dir
    self.root = config.data.dataset_root_dir
    if self.config.data.sampler:
      param = {}
      param['n_points'] = self.config.data.num_sample
      self.sampler = torchtransform.Compose([t.RandomSamplePoints(parameters=param)]) 
    self.files = []
    is_master = du.is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True
    if is_master: logger.info(f"Loading the subset {phase} from {self.root}")
    if phase == "train" or phase == 'train_kmean':
       fname_txt = os.path.join(self.root, self.root_filelist)
       if not os.path.exists(fname_txt): fname_txt = config.data.scannet_match_dir
       with open(fname_txt) as f:
         content = f.readlines()
       fnames = [x.strip().split() for x in content]
       for fname in fnames:
         if len(fname) == 1:
           self.files.append(fname[0])
         else:
          if fname[0] not in self.files:
            self.files.append(fname[0])
          if fname[1] not in self.files:
            self.files.append(fname[1])
    else:
        raise NotImplementedError    
    self.logger.info('train dataset has %d deduplicated data.'%(len(self.files)))

  def __getitem__(self, idx):
    file = os.path.join(self.root, self.files[idx])
    data = np.load(file)
    xyz = data["pcd"]
    xyz = pc_normalize(xyz)
    
    if self.random_rotation:
      T = sample_random_trans(xyz, self.randg, self.rotation_range)
      trans = np.linalg.inv(T)
      xyz = apply_transform(xyz, T)
    else:
      trans = np.identity(4)

    return (xyz, idx)


class ScanNetPseudoPairPointDataset(ScanNetPseudoPointDataset):
  def __init__(self, 
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    super(ScanNetPseudoPairPointDataset, self).__init__(phase, logger, transform, random_rotation, random_scale, manual_seed, config)
    self.ori_transform = torchtransform.Compose([t.PointcloudToTensor()])
    self.rad_transform = torchtransform.Compose([t.PointcloudToTensor(), t.PointcloudScale(scale_low=0.75, scale_high=1.25)])

  def get_ground_label(self, xyz):
    diff = xyz[:,2].max()-xyz[:,2].min()
    ground_label = (xyz[:,2] - diff * 0.02) < 0.1
    ground_label = ground_label.astype(np.int32) 
    return ground_label

  def __getitem__(self, idx):
    file = os.path.join(self.root, self.files[idx])
    data = np.load(file)
    xyz = data["pcd"]
    # compute ground label
    ground_label = self.get_ground_label(xyz)

    xyz_ori = pc_normalize(xyz)
  
    T1 = sample_random_trans(xyz_ori, self.randg, self.rotation_range)
    xyz_rad = apply_transform(xyz_ori, T1)

    T2 = sample_random_trans(xyz_ori, self.randg, self.rotation_range)
    xyz_ori = apply_transform(xyz_ori, T2)

    xyz_rad = self.rad_transform(xyz_rad)
    xyz_ori = self.ori_transform(xyz_ori)

    return (xyz_ori, xyz_rad, idx, ground_label)


class ScanNetAugPairPointDataset(ScanNetPseudoPairPointDataset):
  def __init__(self, 
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    super(ScanNetAugPairPointDataset, self).__init__(phase, logger, transform, random_rotation, random_scale, manual_seed, config)  
    self.aug_crop_num_points = self.config.trainer.aug_crop_num_points
    if not self.config.trainer.xy_crop: 
      centroid_file = self.config.trainer.centroid_file
      centroid = pd.read_csv(centroid_file, sep=' ', header=None).values
      self.random_center = torch.from_numpy(centroid)

  def get_crop_data_index(self, xyz, idx):
    center = self.random_center[idx]
    # compute distance to center
    distance_matrix = torch.norm(center.unsqueeze(0) - xyz, p=2 ,dim=-1)
    dist_idx = torch.argsort(distance_matrix,dim=-1, descending=False)
    # select the nearst points
    sample_index = dist_idx[:self.aug_crop_num_points]
    xyz_sample = xyz[sample_index]
    return xyz_sample, sample_index

  def get_xy_crop_data_index(self, xyz, idx):
    edge = (idx % 2)
    if edge == 0: dist_matrix = torch.tensor(xyz[:,0])
    else: dist_matrix = torch.tensor(xyz[:,1])
    dist_idx = torch.argsort(dist_matrix, dim=0, descending=False)
    sample_index = dist_idx[:self.aug_crop_num_points]
    xyz_sample = xyz[sample_index]
    return xyz_sample, sample_index    

  def __getitem__(self, idx):
    file = os.path.join(self.root, self.files[idx])
    data = np.load(file)
    xyz = data["pcd"]
    
    # compute ground label
    ground_label = self.get_ground_label(xyz)
    
    xyz_ori = pc_normalize(xyz)
    xyz_rad = xyz_ori.copy()
  
    # get crop
    if self.config.trainer.xy_crop: 
      xyz_ori, sample_ori_pindex = self.get_xy_crop_data_index(xyz_ori, idx)
    else: xyz_ori, sample_ori_pindex = self.get_crop_data_index(xyz_ori, idx)

    T1 = sample_random_trans(xyz_rad, self.randg, self.rotation_range)
    xyz_rad = apply_transform(xyz_rad, T1)

    T2 = sample_random_trans(xyz_ori, self.randg, self.rotation_range)
    xyz_ori = apply_transform(xyz_ori, T2)

    xyz_rad = self.rad_transform(xyz_rad)
    xyz_ori = self.ori_transform(xyz_ori)

    ground_label = ground_label[sample_ori_pindex]


    return (xyz_ori, xyz_rad, idx, ground_label, sample_ori_pindex)


class ScanNetGroundRanSacPointDataset(ScanNetRanSacPointDataset):
  def __init__(self, 
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.transform = transform
    self.voxel_size = config.data.voxel_size
    self.matching_search_voxel_size = \
        config.data.voxel_size * config.trainer.positive_pair_search_voxel_size_multiplier
    self.random_scale = random_scale
    self.min_scale = config.trainer.min_scale
    self.max_scale = config.trainer.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.trainer.rotation_range
    self.randg = np.random.RandomState()
    self.logger = logger
    self.config = config
  
    if manual_seed:
      self.reset_seed()
    
    self.root_filelist = config.data.scannet_match_dir
    self.root = config.data.dataset_root_dir
    if self.config.data.sampler:
      param = {}
      param['n_points'] = self.config.data.num_sample
      self.sampler = torchtransform.Compose([t.RandomSamplePoints(parameters=param)]) 
    self.files = []
    is_master = du.is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True
    if is_master: logger.info(f"Loading the subset {phase} from {self.root}")
    if phase == "train":
       fname_txt = os.path.join(self.root, self.root_filelist)
       if not os.path.exists(fname_txt): fname_txt = config.data.scannet_match_dir
       with open(fname_txt) as f:
         content = f.readlines()
       fnames = [x.strip().split() for x in content]
       for fname in fnames:
         if len(fname) == 1:
           self.files.append(fname[0])
         else:
          if fname[0] not in self.files:
            self.files.append(fname[0])
          if fname[1] not in self.files:
            self.files.append(fname[1])
    else:
        raise NotImplementedError    
    self.logger.info('train dataset has %d deduplicated data.'%(len(self.files)))  

    # pseudo label storage
    self.pseudo_label = np.zeros((len(self.files), self.config.data.num_sample))
    self.pseudo_proceed_flag = np.zeros(len(self.files))

  def get_matches(self, xyz_ref, psud_labels, psud_pc_idx, search_voxel_size):
    source_copy = copy.deepcopy(xyz_ref)
    target_copy = copy.deepcopy(xyz_ref)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
      [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
      for j in idx:
        if psud_labels[i] == psud_labels[j] and i != j:
          match_inds.append((psud_pc_idx[i], psud_pc_idx[j]))
    return match_inds

  def get_matches_according_label(self, psud_labels):
    match_inds = []
    num_dict = Counter(psud_labels)
    for cls_id in num_dict:
      if cls_id != -1:
        cls_point_id = np.argwhere(psud_labels == cls_id)
        cls_point_id = cls_point_id.reshape(cls_point_id.shape[0])
        min_point_num = min(cls_point_id.shape[0], 500)
        shuffle_choice = np.random.permutation(np.arange(cls_point_id.shape[0]))
        cls_point_id = cls_point_id[shuffle_choice[:min_point_num]]       
        pair_list = list(itertools.combinations(cls_point_id,2))
        shuffle(pair_list)
        min_sample = min(len(pair_list), self.config.trainer.pair_num_thred)
        for i in range(min_sample): match_inds.append(pair_list[i])
    return match_inds   

  def get_ground_label(self, xyz):
    diff = xyz[:,2].max()-xyz[:,2].min()
    ground_label = (xyz[:,2] - diff * 0.02) < self.config.trainer.floor_thred
    ground_label = ground_label.astype(np.int32) 
    return ground_label

  def raw_dbscan(self, xyz, eps=0.09, min_samples=15):
      clustering = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
      labels = clustering.labels_
      labels_obj_idx = np.argwhere(labels != -1)
      if labels_obj_idx.shape[0] > 0: labels[labels_obj_idx] += 1
      return labels, np.arange(labels.shape[0])

  def __getitem__(self, idx):
    file = os.path.join(self.root, self.files[idx])
    data = np.load(file)
    xyz = data["pcd"]
    
    ground_label = self.get_ground_label(xyz)
    xyz = pc_normalize(xyz)

    xyz_ground = xyz.copy()[ground_label==1]
    xyz_obj = xyz.copy()[ground_label==0]
    xyz = np.concatenate((xyz_obj, xyz_ground), axis=0)

    if self.pseudo_proceed_flag[idx] == 1:
      #matches = self.pseudo_matches[idx]
      psud_labels = self.pseudo_label[idx]
    else:
      # ransac of obj
      db_labels, db_pc_idx = self.raw_dbscan(xyz_obj, eps=0.05, min_samples=15)
      gt_labels = np.zeros(xyz_ground.shape[0])
      gt_idx = np.arange(xyz_obj.shape[0], xyz.shape[0])
      psud_labels = np.concatenate((db_labels, gt_labels), axis=0)
      #psud_idx = np.concatenate((db_pc_idx, gt_idx), axis=0)
      
      # get match label pair
      #pcd = o3d.geometry.PointCloud()
      #pcd.points = o3d.utility.Vector3dVector(xyz)
      #matches = self.get_matches(pcd, psud_labels, psud_idx, self.matching_search_voxel_size)
      #self.pseudo_matches[idx] = matches
      self.pseudo_label[idx] = psud_labels
      self.pseudo_proceed_flag[idx] = 1

    matches = self.get_matches_according_label(psud_labels)

    if self.random_rotation:
      T = sample_random_trans(xyz, self.randg, self.rotation_range)
      trans = np.linalg.inv(T)
      xyz = apply_transform(xyz, T)
    else:
      trans = np.identity(4)

    return (xyz, matches, trans)


class ScanNetDBSCANPairPointDataset(ScanNetGroundRanSacPointDataset):
  def __init__(self, 
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
      self.phase = phase
      self.files = []
      self.transform = transform
      self.voxel_size = config.data.voxel_size
      self.matching_search_voxel_size = \
          config.data.voxel_size * config.trainer.positive_pair_search_voxel_size_multiplier
      self.random_scale = random_scale
      self.min_scale = config.trainer.min_scale
      self.max_scale = config.trainer.max_scale
      self.random_rotation = random_rotation
      self.rotation_range = config.trainer.rotation_range
      self.randg = np.random.RandomState()
      self.logger = logger
      self.config = config
  
      if manual_seed:
        self.reset_seed()
    
      self.root_filelist = config.data.scannet_match_dir
      self.root = config.data.dataset_root_dir
      if self.config.data.sampler:
        param = {}
        param['n_points'] = self.config.data.num_sample
        self.sampler = torchtransform.Compose([t.RandomSamplePoints(parameters=param)]) 
      self.files = []
      is_master = du.is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True
      if is_master: logger.info(f"Loading the subset {phase} from {self.root}")
      if phase == "train" or phase == 'train_kmean':
         fname_txt = os.path.join(self.root, self.root_filelist)
         if not os.path.exists(fname_txt): fname_txt = config.data.scannet_match_dir
         with open(fname_txt) as f:
           content = f.readlines()
         fnames = [x.strip().split() for x in content]
         for fname in fnames:
           if len(fname) == 1:
             self.files.append(fname[0])
           else:
            if fname[0] not in self.files:
              self.files.append(fname[0])
            if fname[1] not in self.files:
              self.files.append(fname[1])
      else:
          raise NotImplementedError    
      self.logger.info('train dataset has %d deduplicated data.'%(len(self.files)))  

      # pseudo label storage
      self.pseudo_label = np.zeros((len(self.files), self.config.data.num_sample))
      self.pseudo_proceed_flag = np.zeros(len(self.files))

      self.ori_transform = torchtransform.Compose([t.PointcloudToTensor()])
      self.rad_transform = torchtransform.Compose([t.PointcloudToTensor(), t.PointcloudScale(scale_low=0.75, scale_high=1.25)])

  def __getitem__(self, idx):
    file = os.path.join(self.root, self.files[idx])
    data = np.load(file)
    xyz = data["pcd"]
    
    ground_label = self.get_ground_label(xyz)
    xyz = pc_normalize(xyz)

    xyz_ground = xyz.copy()[ground_label==1]
    xyz_obj = xyz.copy()[ground_label==0]
    xyz = np.concatenate((xyz_obj, xyz_ground), axis=0)

    if self.pseudo_proceed_flag[idx] == 1:
      psud_labels = self.pseudo_label[idx]
    else:
      # ransac of obj
      db_labels, db_pc_idx = self.raw_dbscan(xyz_obj, eps=self.config.trainer.dbscan.eps, min_samples=self.config.trainer.dbscan.min_sample)
      gt_labels = np.zeros(xyz_ground.shape[0])
      gt_idx = np.arange(xyz_obj.shape[0], xyz.shape[0])
      psud_labels = np.concatenate((db_labels, gt_labels), axis=0)
      self.pseudo_label[idx] = psud_labels
      self.pseudo_proceed_flag[idx] = 1

    matches = self.get_matches_according_label(psud_labels)

    xyz_ori = xyz.copy()
    T1 = sample_random_trans(xyz_ori, self.randg, self.rotation_range)
    xyz_rad = apply_transform(xyz_ori, T1)

    #T2 = sample_random_trans(xyz_ori, self.randg, self.rotation_range)
    #xyz_ori = apply_transform(xyz_ori, T2)

    xyz_rad = self.rad_transform(xyz_rad)
    xyz_ori = self.ori_transform(xyz_ori)

    # assumption: Counter(psud_labels) < 100
    return (xyz_ori, xyz_rad, matches, idx, psud_labels+idx*100)   


# Voxel data loader 
class ScanNetDBSCANPairVoxelDataset(ScanNetDBSCANPairPointDataset):
  def __init__(self, 
               phase,
               logger,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
      super(ScanNetDBSCANPairVoxelDataset, self).__init__(phase, logger, transform, random_rotation, random_scale, manual_seed, config)
      self.voxel_size = config.data.voxel_size

  def align_ori_rad_voxel(self, fea_ori, fea_rad, T1):
    fea_ori_rotate = self.ori_transform(apply_transform(fea_ori.numpy(), T1))
    distance = square_distance_np(fea_ori_rotate, fea_rad)
    M, N = distance.shape
    if M <= N:
      index_ori = torch.arange(0, M)
      index_rad = torch.argmin(distance, dim=1)
      index_set_list = list(set(index_rad))
      assert index_rad.shape[0] == len(index_set_list)
    else:
      index_rad = torch.arange(0, N)
      index_ori = torch.argmin(distance, dim=0)
      index_set_list = list(set(index_ori))
      assert index_ori.shape[0] == len(index_set_list)
    return index_ori, index_rad

  def __getitem__(self, idx):
    use_color = False
    if self.config.data.get('use_color') and self.config.data.use_color: use_color = True
    file = os.path.join(self.root, self.files[idx])
    data = np.load(file)
    xyz = data["pcd"]
    
    # ground points detection
    ground_label = self.get_ground_label(xyz)
    xyz = pc_normalize(xyz)
    xyz_ground = xyz.copy()[ground_label==1]
    xyz_obj = xyz.copy()[ground_label==0]
    xyz = np.concatenate((xyz_obj, xyz_ground), axis=0)

    if use_color:
      color = data['color']
      color_ground = color.copy()[ground_label==1]
      color_obj = color.copy()[ground_label==0]
      color = np.concatenate((color_obj, color_ground), axis=0)

    # DBSCAN obtain pseudo labels
    if self.pseudo_proceed_flag[idx] == 1:
      psud_labels = self.pseudo_label[idx]
    else:
      # ransac of obj
      db_labels, db_pc_idx = self.raw_dbscan(xyz_obj, eps=self.config.trainer.dbscan.eps, min_samples=self.config.trainer.dbscan.min_sample)
      gt_labels = np.zeros(xyz_ground.shape[0])
      gt_idx = np.arange(xyz_obj.shape[0], xyz.shape[0])
      psud_labels = np.concatenate((db_labels, gt_labels), axis=0)
      self.pseudo_label[idx] = psud_labels
      self.pseudo_proceed_flag[idx] = 1

    # two views augmentation
    xyz_ori = xyz.copy()
    T1 = sample_random_trans(xyz_ori, self.randg, self.rotation_range)
    xyz_rad = apply_transform(xyz_ori, T1)
    xyz_rad = self.ori_transform(xyz_rad) # toTensor (there should not be jittering)
    xyz_ori = self.ori_transform(xyz_ori) # toTensor
    
    # label to tensor
    tpsud_labels = torch.from_numpy(psud_labels).int()

    # voxeliazation
    ori_index, xyz_ori_psud_label = ME.utils.sparse_quantize(xyz_ori / self.voxel_size, labels=tpsud_labels, return_index=True)
    rad_index, xyz_rad_psud_label = ME.utils.sparse_quantize(xyz_rad / self.voxel_size, labels=tpsud_labels, return_index=True)
    # BUG need to be fixed in the sampling 
    #sel_ori =  ME.utils.sparse_quantize(xyz_ori / self.voxel_size, return_index=True)

    # --------- #
    pcd_ori = make_open3d_point_cloud(xyz_ori)
    pcd_rad = make_open3d_point_cloud(xyz_rad)
    pcd_ori.points = o3d.utility.Vector3dVector(np.array(pcd_ori.points)[ori_index])
    pcd_rad.points = o3d.utility.Vector3dVector(np.array(pcd_rad.points)[rad_index])
    xyzori = np.array(pcd_ori.points)
    xyzrad = np.array(pcd_rad.points)
    xyz_ori_voxel = np.floor(xyzori / self.voxel_size)
    xyz_rad_voxel = np.floor(xyzrad / self.voxel_size)    
    # --------- #

    # ******  #
    #xyz_ori_voxel = torch.floor(xyz_ori[ori_index] / self.voxel_size)
    #xyz_rad_voxel = torch.floor(xyz_rad[rad_index] / self.voxel_size)
    # ******  #
    
    if use_color:
      fea_ori = torch.tensor(color[ori_index])
      fea_rad = torch.tensor(color[rad_index])
    else:
      fea_ori = xyz_ori[ori_index]
      fea_rad = xyz_rad[rad_index]

    pc_ori = xyz_ori[ori_index]
    pc_rad = xyz_rad[rad_index]

    ori_psud_label = psud_labels[ori_index]
    rad_psud_label = psud_labels[rad_index]

    # obtain matches 
    ori_matches = self.get_matches_according_label(ori_psud_label)
    rad_matches = self.get_matches_according_label(rad_psud_label)


    # Should DE-BUG for this place and align_ori_rad_voxel function
    index_ori, index_rad = self.align_ori_rad_voxel(pc_ori, pc_rad, T1)

    # assumption: Counter(psud_labels) < 100
    return (fea_ori, fea_rad, xyz_ori_voxel, xyz_rad_voxel, ori_matches, rad_matches, \
            index_ori, index_rad, idx, ori_psud_label+idx*100, rad_psud_label+idx*100)