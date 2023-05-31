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
import dataset.transforms as t
import util.distributed as du
import open3d as o3d
from random import shuffle
from scipy.linalg import expm, norm
from torchvision import transforms as torchtransform
from sklearn import cluster
from collections import Counter
from .data_util import *


class ScanNetPointsDataset:
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

  def reset_seed(self, seed=0):
    self.logger.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def __len__(self):
    return len(self.files)

  def dbscan(self, xyz, eps=0.09, min_samples=15):
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
    
    ground_label = self.get_ground_label(xyz)
    xyz = pc_normalize(xyz)

    xyz_ground = xyz.copy()[ground_label==1]
    xyz_obj = xyz.copy()[ground_label==0]
    xyz = np.concatenate((xyz_obj, xyz_ground), axis=0)

    if self.pseudo_proceed_flag[idx] == 1:
      psud_labels = self.pseudo_label[idx]
    else:
      # -------------- DBSCAN to obtain instance masks ----------------
      # density-based clustering to obtain instance mask label
      db_labels, db_pc_idx = self.dbscan(xyz_obj, eps=self.config.trainer.dbscan.eps, min_samples=self.config.trainer.dbscan.min_sample)
      # the points on the floor (z == 0)
      gt_labels = np.zeros(xyz_ground.shape[0])
      gt_idx = np.arange(xyz_obj.shape[0], xyz.shape[0])
      # concatenate instance label and floor label
      psud_labels = np.concatenate((db_labels, gt_labels), axis=0)
      # record the label
      self.pseudo_label[idx] = psud_labels
      self.pseudo_proceed_flag[idx] = 1
      # --------------------------------------------------------------


    matches = self.get_matches_according_label(psud_labels)

    xyz_ori = xyz.copy()
    T1 = sample_random_trans(xyz_ori, self.randg, self.rotation_range)
    xyz_rad = apply_transform(xyz_ori, T1)

    xyz_rad = self.rad_transform(xyz_rad)
    xyz_ori = self.ori_transform(xyz_ori)

    # assumption: Counter(psud_labels) < 100
    return (xyz_ori, xyz_rad, matches, idx, psud_labels+idx*100)   
