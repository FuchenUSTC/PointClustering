# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2022/01/01
# desc: dataset loader

import torch
import torch.utils.data
import dataset.transforms as t
from .data_sampler import DistributedInfSampler
from .data_collate import *
from .pretrain_data_loader import *
from .scannet import *


ALL_DATASETS = [ScanNetPointsDataset, ScanNetDBSCANPairPointDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, batch_size, logger, phase='train', num_threads=0):

  if config.data.dataset not in dataset_str_mapping.keys():
    logger.error(f'Dataset {config.data.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.data.dataset]

  transforms = []
  use_random_rotation = config.trainer.use_random_rotation
  use_random_scale = config.trainer.use_random_scale
  transforms += [t.Jitter()]

  dset = Dataset(
      phase=phase,
      transform=t.Compose(transforms),
      random_scale=use_random_scale,
      random_rotation=use_random_rotation,
      config=config,
      logger=logger)
  if config.data.dataset == 'ScanNetPointDataset':
    collate_pair_fn = scannetpoint_for_seg_collate_fn
  elif config.data.dataset == 'ScanNetDBSCANPairPointDataset':
    collate_pair_fn = scannetpoint_dbscan_pair_collate_fn
  batch_size = batch_size // config.misc.num_gpus

  if config.misc.num_gpus > 1:
    if phase == 'train' or phase == 'trainval': sampler = DistributedInfSampler(dset)
    else: sampler = DistributedInfSampler(dset, shuffle=False)
  else:
    sampler = None
  
  # sampler convert 
  if ('ShapeNetPartSeg' in config.data.dataset or 'PartNetSeg' in config.data.dataset) \
  and (phase == 'val' or phase == 'test'): sampler = None 

  # only val in one gpu
  if 'Stanford3D' in config.data.dataset and phase == 'val': sampler = None
  if 'StanfordVoxel' in config.data.dataset and phase == 'val': batch_size = 1
  if 'ScanNetVoxel' in config.data.dataset and phase == 'val':
    sampler = None
    batch_size = 1
  if 'ScanNetPointDataset' in config.data.dataset and phase == 'val': sampler = None
  if 'ScanNetPointWholeDataset' in config.data.dataset and phase == 'val':
    sampler = None
    batch_size = 1
  if 'LinearSVMTrainer' in config.trainer.trainer: sampler = None


  if phase == 'train' or phase == 'trainval':
    drop_last_ = True
  else: drop_last_ = False

  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_threads,
      collate_fn=collate_pair_fn,
      pin_memory=True,
      sampler=sampler,
      drop_last=drop_last_)

  return loader
