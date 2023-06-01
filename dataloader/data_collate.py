# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2022/05/01
# desc: dataset collate

import torch 
import numpy as np


def default_collate_pair_fn(list_data):
  xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
      zip(*list_data))
  xyz_batch0, coords_batch0, feats_batch0 = [], [], []
  xyz_batch1, coords_batch1, feats_batch1 = [], [], []
  matching_inds_batch, trans_batch, len_batch = [], [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(coords0):

    N0 = coords0[batch_id].shape[0]
    N1 = coords1[batch_id].shape[0]

    # Move batchids to the beginning
    xyz_batch0.append(torch.from_numpy(xyz0[batch_id]))
    coords_batch0.append(
        torch.cat((torch.ones(N0, 1).int() * batch_id, 
                   torch.from_numpy(coords0[batch_id]).int()), 1))
    feats_batch0.append(torch.from_numpy(feats0[batch_id]))

    xyz_batch1.append(torch.from_numpy(xyz1[batch_id]))
    coords_batch1.append(
        torch.cat((torch.ones(N1, 1).int() * batch_id, 
                   torch.from_numpy(coords1[batch_id]).int()), 1))
    feats_batch1.append(torch.from_numpy(feats1[batch_id]))

    trans_batch.append(torch.from_numpy(trans[batch_id]))
    
    # in case 0 matching
    if len(matching_inds[batch_id]) == 0:
      matching_inds[batch_id].extend([0, 0])
    
    matching_inds_batch.append(
        torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])

    # Move the head
    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  # Concatenate all lists
  xyz_batch0 = torch.cat(xyz_batch0, 0).float()
  coords_batch0 = torch.cat(coords_batch0, 0).int()
  feats_batch0 = torch.cat(feats_batch0, 0).float()
  xyz_batch1 = torch.cat(xyz_batch1, 0).float()
  coords_batch1 = torch.cat(coords_batch1, 0).int()
  feats_batch1 = torch.cat(feats_batch1, 0).float()
  trans_batch = torch.cat(trans_batch, 0).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
  return {
      'pcd0': xyz_batch0,
      'pcd1': xyz_batch1,
      'sinput0_C': coords_batch0,
      'sinput0_F': feats_batch0,
      'sinput1_C': coords_batch1,
      'sinput1_F': feats_batch1,
      'correspondences': matching_inds_batch,
      'T_gt': trans_batch,
      'len_batch': len_batch,
  }


def scannetpoint_collate_fn(list_data):
  xyz0, xyz1, matching_inds, trans = list(zip(*list_data))
  xyz_batch0, xyz_batch1 = [], []
  matching_inds_batch, trans_batch, len_batch = [], [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(xyz0):
    N0 = xyz0[batch_id].shape[0]
    N1 = xyz1[batch_id].shape[0]
  
    xyz_batch0.append((xyz0[batch_id]))
    xyz_batch1.append((xyz1[batch_id]))
    trans_batch.append((trans[batch_id]))
    
    if len(matching_inds[batch_id]) == 0: matching_inds[batch_id].extend([0, 0])
    matching_inds_batch.append(torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])

    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  xyz_batch0 = torch.Tensor(np.array(xyz_batch0)).float()
  xyz_batch1 = torch.Tensor(np.array(xyz_batch1)).float()
  trans_batch = torch.Tensor(np.array(trans_batch)).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
  
  return {
      'pcd0': xyz_batch0,
      'pcd1': xyz_batch1,
      'correspondences': matching_inds_batch,
      'T_gt': trans_batch,
      'len_batch': len_batch,
  }  


def scannetpoint_ransac_collate_fn(list_data):
  xyz, matching_inds, trans = list(zip(*list_data))
  xyz_batch = []
  matching_inds_batch, trans_batch, len_batch = [], [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(xyz):
    N0 = xyz[batch_id].shape[0]
    N1 = xyz[batch_id].shape[0]
  
    xyz_batch.append((xyz[batch_id]))
    trans_batch.append((trans[batch_id]))
    
    if len(matching_inds[batch_id]) == 0: matching_inds[batch_id].extend([0, 0])
    matching_inds_batch.append(torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])

    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  xyz_batch = torch.Tensor(xyz_batch).float()
  trans_batch = torch.Tensor(trans_batch).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
  
  return {
      'pcd': xyz_batch,
      'correspondences': matching_inds_batch,
      'T_gt': trans_batch,
  }  


def scannetpoint_pseudo_collate_fn(list_data):
  xyz, idx = list(zip(*list_data))
  xyz_batch, idx_batch = [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(xyz):
    N0 = xyz[batch_id].shape[0]
    N1 = xyz[batch_id].shape[0]
  
    xyz_batch.append((xyz[batch_id]))
    idx_batch.append((idx[batch_id]))
  

  xyz_batch = torch.Tensor(np.array(xyz_batch)).float()
  idx_batch = torch.Tensor(np.array(idx_batch)).int()

  return {
      'pcd': xyz_batch,
      'idx': idx_batch,
  }  


def scannetpoint_pseudo_pair_collate_fn(list_data):
  xyz_ori, xyz_rad, idx, ground_label = list(zip(*list_data))
  xyz_ori_batch, xyz_rad_batch, idx_batch, ground_label_batch = [], [], [], []

  for batch_id, _ in enumerate(xyz_ori):
    xyz_ori_batch.append(np.array(xyz_ori[batch_id]))
    xyz_rad_batch.append(np.array(xyz_rad[batch_id]))
    idx_batch.append((idx[batch_id]))
    ground_label_batch.append(ground_label[batch_id])
  
  xyz_ori_batch = torch.tensor(xyz_ori_batch).float()
  xyz_rad_batch = torch.tensor(xyz_rad_batch).float()
  idx_batch = torch.Tensor(np.array(idx_batch)).int()
  ground_label_batch = torch.tensor(ground_label_batch).int()

  return {
      'pcd_ori': xyz_ori_batch,
      'pcd_rad': xyz_rad_batch,
      'idx': idx_batch,
      'ground_label': ground_label_batch,
  }  


def scannetpoint_aug_pair_collate_fn(list_data):
  xyz_ori, xyz_rad, idx, ground_label, sample_ori_pindex = list(zip(*list_data))
  xyz_ori_batch, xyz_rad_batch, idx_batch, ground_label_batch, sample_ori_pindex_batch = [], [], [], [], []

  for batch_id, _ in enumerate(xyz_ori):
    xyz_ori_batch.append(np.array(xyz_ori[batch_id]))
    xyz_rad_batch.append(np.array(xyz_rad[batch_id]))
    idx_batch.append((idx[batch_id]))
    ground_label_batch.append(ground_label[batch_id])
    sample_ori_pindex_batch.append(np.array(sample_ori_pindex[batch_id]))
  
  xyz_ori_batch = torch.tensor(xyz_ori_batch).float()
  xyz_rad_batch = torch.tensor(xyz_rad_batch).float()
  idx_batch = torch.Tensor(np.array(idx_batch)).int()
  ground_label_batch = torch.tensor(ground_label_batch).int()
  sample_ori_pindex_batch = torch.tensor(sample_ori_pindex_batch).long()

  return {
      'pcd_ori': xyz_ori_batch,
      'pcd_rad': xyz_rad_batch,
      'idx': idx_batch,
      'ground_label': ground_label_batch,
      'sample_ori_pindex': sample_ori_pindex_batch,
  }  


def scannetpoint_dbscan_pair_collate_fn(list_data):
  xyz_ori, xyz_rad, matching_inds, idx, pseudo_labels = list(zip(*list_data))
  xyz_ori_batch, xyz_rad_batch, matching_inds_batch, idx_batch, pseudo_labels_batch = [], [], [], [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(xyz_ori):
    N0 = xyz_ori[batch_id].shape[0]
    N1 = xyz_ori[batch_id].shape[0]

    xyz_ori_batch.append(np.array(xyz_ori[batch_id]))
    xyz_rad_batch.append(np.array(xyz_rad[batch_id]))
    idx_batch.append((idx[batch_id]))
    pseudo_labels_batch.append(pseudo_labels[batch_id])

    if len(matching_inds[batch_id]) == 0: matching_inds[batch_id].extend([0, 0])
    matching_inds_batch.append(torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  
  xyz_ori_batch = torch.tensor(xyz_ori_batch).float()
  xyz_rad_batch = torch.tensor(xyz_rad_batch).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
  idx_batch = torch.Tensor(np.array(idx_batch)).int()
  pseudo_labels_batch = torch.tensor(pseudo_labels_batch).int()

  return {
      'pcd_ori': xyz_ori_batch,
      'pcd_rad': xyz_rad_batch,
      'correspondences': matching_inds_batch,
      'idx': idx_batch,
      'pseudo_labels': pseudo_labels_batch,
  }  


def scannetvoxel_dbscan_pair_collate_fn(list_data):
  fea_ori, fea_rad, xyz_ori_voxel, xyz_rad_voxel, ori_matching_inds, rad_matching_inds, \
  ori_index, rad_index, idx, ori_psud_label, rad_psud_label = list(zip(*list_data))
  
  fea_ori_batch, fea_rad_batch, xyz_ori_batch, xyz_rad_batch, \
  ori_matching_inds_batch, rad_matching_inds_batch, ori_index_batch, rad_index_batch, \
  idx_batch, ori_psud_label_batch, rad_psud_label_batch = [], [], [], [], [], [], [], [], [], [], []
  
  ori_cut_id_batch = []
  rad_cut_id_batch = []

  batch_id = 0
  ori_cut_id, rad_cut_id = 0, 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(fea_ori):
    # N0 maynot equal to N1
    N0 = fea_ori[batch_id].shape[0]
    N1 = fea_rad[batch_id].shape[0]

    fea_ori_batch.append(fea_ori[batch_id])
    fea_rad_batch.append(fea_rad[batch_id])

    xyz_ori_batch.append(torch.cat((torch.ones(N0, 1).int() * batch_id, torch.from_numpy(xyz_ori_voxel[batch_id]).int()), 1))
    xyz_rad_batch.append(torch.cat((torch.ones(N1, 1).int() * batch_id, torch.from_numpy(xyz_rad_voxel[batch_id]).int()), 1))

    ori_index_batch.append(ori_index[batch_id])
    rad_index_batch.append(rad_index[batch_id])
    
    ori_psud_label_batch.append(torch.tensor(ori_psud_label[batch_id]))
    rad_psud_label_batch.append(torch.tensor(rad_psud_label[batch_id]))

    idx_batch.append((idx[batch_id]))

    if len(ori_matching_inds[batch_id]) == 0: ori_matching_inds[batch_id].extend([0, 0])
    if len(rad_matching_inds[batch_id]) == 0: rad_matching_inds[batch_id].extend([0, 0])
    ori_matching_inds_batch.append(torch.from_numpy(np.array(ori_matching_inds[batch_id]) + curr_start_inds)) 
    rad_matching_inds_batch.append(torch.from_numpy(np.array(rad_matching_inds[batch_id]) + curr_start_inds)) 

    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

    ori_cut_id += N0
    rad_cut_id += N1
    
    ori_cut_id_batch.append(ori_cut_id)
    rad_cut_id_batch.append(rad_cut_id)


  fea_ori_batch = torch.cat(fea_ori_batch, 0).float()
  fea_rad_batch = torch.cat(fea_rad_batch, 0).float()
  xyz_ori_batch = torch.cat(xyz_ori_batch, 0).int()
  xyz_rad_batch = torch.cat(xyz_rad_batch, 0).int()
  ori_index_batch = torch.cat(ori_index_batch, 0).int()
  rad_index_batch = torch.cat(rad_index_batch, 0).int()
  ori_psud_label_batch = torch.cat(ori_psud_label_batch).int()
  rad_psud_label_batch = torch.cat(rad_psud_label_batch).int()
  ori_matching_inds_batch = torch.cat(ori_matching_inds_batch, 0).int()
  rad_matching_inds_batch = torch.cat(rad_matching_inds_batch, 0).int()
  ori_cut_id_batch = torch.Tensor(np.array(ori_cut_id_batch)).int()
  rad_cut_id_batch = torch.Tensor(np.array(rad_cut_id_batch)).int()
  idx_batch = torch.Tensor(np.array(idx_batch)).int()

  return {
      'fea_ori': fea_ori_batch,
      'fea_rad': fea_rad_batch,
      'pcd_ori': xyz_ori_batch,
      'pcd_rad': xyz_rad_batch,
      'ori_index': ori_index_batch,
      'rad_index': rad_index_batch,
      'ori_psud_label': ori_psud_label_batch,
      'rad_psud_label': rad_psud_label_batch,
      'ori_cut_id': ori_cut_id_batch,
      'rad_cut_id': rad_cut_id_batch,
      'ori_correspondences': ori_matching_inds_batch,
      'rad_correspondences': rad_matching_inds_batch,
      'idx': idx_batch,
  }   


def modelnet_collate_fn(list_data):
  xyz, coords, feats, labels = list(zip(*list_data))  
  xyz_batch, coords_batch, feats_batch, label_batch = [], [], [], []

  batch_id = 0
  for batch_id, _ in enumerate(coords):
    N = coords[batch_id].shape[0]
    xyz_batch.append(torch.from_numpy(xyz[batch_id]))
    coords_batch.append(torch.cat((torch.ones(N, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
    feats_batch.append(torch.from_numpy(feats[batch_id]))
    label_batch.append(torch.from_numpy(labels[batch_id]))   

  xyz_batch = torch.cat(xyz_batch, 0).float()
  coords_batch = torch.cat(coords_batch, 0).int()
  feats_batch = torch.cat(feats_batch, 0).float()
  label_batch = torch.tensor(label_batch,dtype=torch.long)
  
  return {
    'pcd': xyz_batch,
    'sinput_C': coords_batch,
    'sinput_F': feats_batch,
    'label': label_batch,
  } 


def modelnetpoint_collate_fn(list_data):
  xyz, labels = list(zip(*list_data))  
  xyz_batch, label_batch = [], []

  batch_id = 0
  for batch_id, _ in enumerate(xyz):
    xyz_batch.append(xyz[batch_id].tolist())
    label_batch.append(labels[batch_id].tolist())   

  xyz_batch = torch.tensor(xyz_batch, dtype=torch.float)
  label_batch = torch.tensor(label_batch, dtype=torch.long)
  
  return {
    'pcd': xyz_batch,
    'label': label_batch,
  } 


def shapenetpart_collate_fn(list_data):
  xyz, coords, feats, mask, point_labels, labels = list(zip(*list_data))  
  xyz_batch, coords_batch, feats_batch, mask_batch, point_label_batch, label_batch = [], [], [], [], [], []

  batch_id = 0
  for batch_id, _ in enumerate(coords):
    N = coords[batch_id].shape[0]
    xyz_batch.append(torch.from_numpy(xyz[batch_id]))
    coords_batch.append(torch.cat((torch.ones(N, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
    feats_batch.append(torch.from_numpy(feats[batch_id]))
    mask_batch.append(torch.from_numpy(mask[batch_id]))
    point_label_batch.append(torch.from_numpy(point_labels[batch_id]))
    label_batch.append(torch.from_numpy(labels[batch_id]))   

  xyz_batch = torch.cat(xyz_batch, 0).float()
  coords_batch = torch.cat(coords_batch, 0).int()
  feats_batch = torch.cat(feats_batch, 0).float()
  mask_batch = torch.cat(mask_batch, 0).float()
  point_label_batch = torch.cat(point_label_batch, 0).long()
  label_batch = torch.tensor(label_batch,dtype=torch.long)
  
  return {
    'pcd': xyz_batch,
    'sinput_C': coords_batch,
    'sinput_F': feats_batch,
    'mask': mask_batch,
    'point_label': point_label_batch,
    'label': label_batch,
  }   


def shapenetpartpoint_collate_fn(list_data):
  xyz, mask, point_labels, labels = list(zip(*list_data))  
  xyz_batch, mask_batch, point_label_batch, label_batch = [], [], [], []
  batch_id = 0
  for batch_id, _ in enumerate(xyz):
    N = xyz[batch_id].shape[0]
    xyz_batch.append(xyz[batch_id].tolist())
    mask_batch.append(mask[batch_id].tolist())
    point_label_batch.append(point_labels[batch_id].tolist())
    label_batch.append(labels[batch_id].tolist())   

  xyz_batch = torch.tensor(xyz_batch, dtype=torch.float)
  mask_batch = torch.tensor(mask_batch, dtype=torch.float)
  point_label_batch = torch.tensor(point_label_batch, dtype=torch.long)
  label_batch = torch.tensor(label_batch,dtype=torch.long)
  
  return {
    'pcd': xyz_batch,
    'mask': mask_batch,
    'point_label': point_label_batch,
    'label': label_batch,
  } 


def stanford3dpoint_collate_fn(list_data):
  xyz, mask, point_labels, cloud_index, input_index = list(zip(*list_data))
  xyz_batch, mask_batch, point_label_batch, cloud_index_batch, input_index_batch = [], [], [], [], []
  batch_id = 0
  for batch_id, _ in enumerate(xyz):
    xyz_batch.append(xyz[batch_id].tolist())
    mask_batch.append(mask[batch_id].tolist())
    point_label_batch.append(point_labels[batch_id].tolist())
    cloud_index_batch.append(cloud_index[batch_id].tolist()) 
    input_index_batch.append(input_index[batch_id].tolist())

  xyz_batch = torch.tensor(xyz_batch, dtype=torch.float)
  mask_batch = torch.tensor(mask_batch, dtype=torch.float)
  point_label_batch = torch.tensor(point_label_batch, dtype=torch.long)
  cloud_index_batch = torch.tensor(cloud_index_batch,dtype=torch.long)
  input_index_batch = torch.tensor(input_index_batch,dtype=torch.long)
  
  return {
    'pcd': xyz_batch,
    'mask': mask_batch,
    'point_label': point_label_batch,
    'cloud_index': cloud_index_batch,
    'input_index': input_index_batch,
  }   


def stanfordvoxel_collate_fn(list_data):
  coords, feats, labels = list(zip(*list_data))
  coords_batch, feats_batch, labels_batch = [], [], []
  batch_id = 0
  batch_num_points = 0
  for batch_id, _ in enumerate(coords):
    num_points = coords[batch_id].shape[0]
    batch_num_points += num_points
    coords_batch.append(
        torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(
        coords[batch_id]).int()), 1))
    feats_batch.append(torch.from_numpy(feats[batch_id]))
    labels_batch.append(torch.from_numpy(labels[batch_id]).int())
    batch_id += 1

  # Concatenate all lists
  coords_batch = torch.cat(coords_batch, 0).int()
  feats_batch = torch.cat(feats_batch, 0).float()
  labels_batch = torch.cat(labels_batch, 0).int()
  return {
    'coords': coords_batch,
    'feats': feats_batch,
    'labels': labels_batch, 
  }


def scannetpoint_for_seg_collate_fn(list_data):
  point_set, semantic_seg, sample_weight, fetch_time = list(zip(*list_data))
  point_set = torch.FloatTensor(point_set)
  semantic_seg = torch.LongTensor(semantic_seg)
  sample_weight = torch.FloatTensor(sample_weight)
  #coords = point_set[:, :, :3]
  #feats = point_set[:, :, 3:]
  return {
      'pcd': point_set,                   # (B, N, 3+c) if whole: (B, K, N, 3+c)
      #'fea': feats,                      # (B, N, 3)
      'point_label': semantic_seg,        # (B, N)
      'sample_weight': sample_weight,     # (B, N)
      'fetch_time': sum(fetch_time)       # float
  }


def ue4point_collate_fn(list_data):
  xyz, obj_idx = list(zip(*list_data))
  xyz_batch, obj_idx_batch, obj_len_batch = [], [], []

  batch_id = 0
  for batch_id, _ in enumerate(xyz):
    N0 = xyz[batch_id].shape[0]
    xyz_batch.append((xyz[batch_id]))
    if len(obj_idx[batch_id]) == 0: obj_idx[batch_id].extend([0, 0])
    obj_idx_batch.append(torch.from_numpy(np.array(obj_idx[batch_id])))
    obj_len_batch.append(len(obj_idx[batch_id]))

  xyz_batch = torch.Tensor(xyz_batch).float()
  obj_idx_batch = torch.cat(obj_idx_batch, 0).int()
  
  return {
      'pcd': xyz_batch,
      'obj_idx': obj_idx_batch,
      'obj_idx_len': obj_len_batch,
  }    
