# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2022/01/08
# desc: train file for PointClustering

import open3d as o3d  # prevent loading error

import argparse
import sys
import os
import json
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from omegaconf import OmegaConf
from easydict import EasyDict as edict

import util.multiprocessing as mpu
from util.logger import setup_logger

from dataloader.data_loaders import make_data_loader
from trainer import *


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


def get_trainer(trainer):
  if trainer == 'PointClusteringTrainer':
    return PointClusteringTrainer
  else:
    raise ValueError(f'Trainer {trainer} not found')
  
  
  # ---- Not included ---- 
  #elif trainer == 'HardestContrastiveLossTrainer':
  #  return HardestContrastiveLossTrainer
  #elif trainer == 'PointNCELossTrainer':
  #  return PointNCELossTrainer
  #elif trainer == 'PointNCELossRanSacTrainer':
  #  return PointNCELossRanSacTrainer
  #elif trainer == 'PointNCEClusterTrainer':
  #  return PointNCEClusterTrainer
  #elif trainer == 'PointClusterSegTrainer':
  #  return PointClusterSegTrainer
  #elif trainer == 'PointClusterPairSegTrainer':
  #  return PointClusterPairSegTrainer
  #elif trainer == 'PointClusterPairAugCropSegTrainer':
  #  return PointClusterPairAugCropSegTrainer
  #elif trainer == 'PointClusterPairDBSCANTrainer':
  #  return PointClusterPairDBSCANTrainer
  #elif trainer == 'PointProtoDBSCANTrainer':
  #  return PointProtoDBSCANTrainer
  #elif trainer == 'PointProtoCoDBSCANTrainer':
  #  return PointProtoCoDBSCANTrainer
  #elif trainer == 'PointProtoCoDBSCANTrainerV2':
  #  return PointProtoCoDBSCANTrainerV2
  #elif trainer == 'PointProtoCoDBSCANCrossTrainer':
  #  return PointProtoCoDBSCANCrossTrainer
  #elif trainer == 'VoxelProtoCoDBSCANTrainer':
  #  return VoxelProtoCoDBSCANTrainer
  #elif trainer == 'VoxelProtoCoDBSCANCrossTrainer':
  #  return VoxelProtoCoDBSCANCrossTrainer
  #elif trainer == 'PointInsNCETrainer':
  #  return PointInsNCETrainer
  #elif trainer == 'SoftmaxLossTrainer':
  #  return SoftmaxLossTrainer
  #elif trainer == 'LinearSVMTrainer':
  #  return LinearSVMTrainer
  #elif trainer == 'MultiShapeCrossEntropyLossTrainer':
  #  return MultiShapeCrossEntropyLossTrainer
  #elif trainer == 'MaskedCrossEntropyLossTrainer':
  #  return MaskedCrossEntropyLossTrainer
  #elif trainer == 'WeightedCrossEntropyLossTrainer':
  #  return WeightedCrossEntropyLossTrainer
  #elif trainer == 'VoxelCrossEntropyLossTrainer':
  #  return VoxelCrossEntropyLossTrainer
  #elif trainer == 'ChamferDistanceLossTrainer':
  #  return ChamferDistanceLossTrainer
  #elif trainer == 'ChamferDistanceLossUE4Trainer':
  #  return ChamferDistanceLossUE4Trainer
  #elif trainer == 'ChamferDistanceLossShapeNet55Trainer':
  #  return ChamferDistanceLossShapeNet55Trainer
  #elif trainer == 'ChamferDistanceRawPointsCompletor':
  #  return ChamferDistanceRawPointsCompletor



def parse_option():
  parser = argparse.ArgumentParser('training')
  parser.add_argument('--config_file', type=str, required=True, help='path of config file (yaml)')
  parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
  args = parser.parse_args()

  if args.config_file:
    config = OmegaConf.load(args.config_file)
  return args, config


def main(config, logger):
  train_loader = make_data_loader(
      config,
      config.trainer.batch_size,
      phase = config.trainer.train_phase,
      num_threads=config.misc.num_workers,
      logger=logger)
  Trainer = get_trainer(config.trainer.trainer)
  if config.pretrain:
    trainer = Trainer(
        config=config,
        data_loader=train_loader,
        logger=logger,)
  else:
    val_loader = make_data_loader(
      config,
      config.trainer.batch_size,
      phase=config.trainer.val_phase,
      num_threads=config.misc.num_workers,
      logger=logger)
    trainer = Trainer(
      config=config,
      data_loader = train_loader,
      val_data_loader = val_loader,
      logger=logger,)
  trainer.train()


if __name__ == "__main__":
  opt, config = parse_option()
  torch.cuda.set_device(opt.local_rank)
  torch.distributed.init_process_group(backend='nccl', init_method='env://')
  cudnn.benchmark = True  

  os.makedirs(config.misc.output_dir, exist_ok=True)
  logger = setup_logger(output=config.misc.output_dir, distributed_rank=dist.get_rank(), name="dbpcl")
  if dist.get_rank() == 0:
      path = os.path.join(config.misc.output_dir, "train_val_3d.config.json")
      with open(path, 'w') as f:
          json.dump(vars(opt), f, indent=2)
      logger.info("Full config saved to {}".format(path))

  main(config, logger)

