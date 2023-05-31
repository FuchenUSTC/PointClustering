import os
import time
import os.path as osp
import numpy as np
from omegaconf import OmegaConf
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.serialization import default_restore_location
from tensorboardX import SummaryWriter

from cluster.batch_kmeans import *
from model import load_model
from util.lr_scheduler import get_optimizer, get_scheduler
from .trainer_util import *

torch.autograd.set_detect_anomaly(True)

 
class PointClusteringTrainer:
  def __init__(
      self,
      config,
      data_loader,
      val_data_loader,
      logger):
    self.stat_freq = config.trainer.stat_freq
    self.resume = False
    assert config.misc.use_gpu and torch.cuda.is_available(), "DDP mode must support GPU"
    num_feats = 3  # always 3 for finetuning.

    self.is_master = is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True
    self.cur_device = torch.cuda.current_device()

    # Model initialization
    Model = load_model(config.net.model)
    model = Model(num_feats, config.net.model_n_out, config, D=3)
    logger.info(model)
    model = model.cuda(device=self.cur_device)
    if config.misc.num_gpus > 1: 
        model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[self.cur_device], output_device=self.cur_device, broadcast_buffers=False,)

    self.config = config
    self.model = model
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.total_iter = 0

    self.optimizer = get_optimizer(model.parameters(), config)
    self.n_iter_per_epoch = len(self.data_loader)
    self.scheduler = get_scheduler(self.optimizer, config, self.n_iter_per_epoch, self.model)
    self._check_load_and_resume()

    if self.is_master:
        self.writer = SummaryWriter(logdir='%s/logs'%(self.config.misc.output_dir))
        if not os.path.exists('%s/weights'%(self.config.misc.output_dir)):
          os.makedirs('%s/weights'%(self.config.misc.output_dir), mode=0o755)
        OmegaConf.save(config, '%s/config.yaml'%(self.config.misc.output_dir))
    
    self.logger = logger

    self.train_kmean_data_loader = val_data_loader
    self.centroid_ori = np.zeros((config.data.cluster_k, config.net.model_n_out))
    self.cluster_label_ori = np.zeros((config.data.num_data, config.data.num_sample))
    self.centroid_rad = np.zeros((config.data.cluster_k, config.net.model_n_out))
    self.cluster_label_rad = np.zeros((config.data.num_data, config.data.num_sample))
    self.criterion = torch.nn.CrossEntropyLoss().cuda()
    
    # train for nce and segmentation
    self.idx_proto_pos = {}
    self.cluster_update_freq = self.config.trainer.cluster_update_freq
    self.begin_clustering_epoch = self.config.trainer.begin_clustering_epoch
    self.kmeans_n_iter = self.config.data.kmeans_n_iter
    self.instance_cluster_k = self.config.data.cluster_k

  def _check_load_and_resume(self):
    # load a model from the given path
    if self.config.misc.weight:
        if self.is_master:
          self.logger.info('===> Loading weights: ' + self.config.misc.weight)
        state = torch.load(self.config.misc.weight, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        load_state(self.model, state['state_dict'], self.logger, self.config.misc.lenient_weight_loading)
        if self.is_master:
          self.logger.info('===> Loaded weights: ' + self.config.misc.weight)
    # resume from the checkpoint if the weight.pth exists
    checkpoint_fn = '%s/weights/weights.pth'%(self.self.config.misc.output_dir)
    if osp.isfile(checkpoint_fn):
      if self.is_master:
        self.logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn, map_location=lambda s, l: default_restore_location(s, 'cpu'))
      self.total_iter = state['curr_iter']
      load_state(self.model, state['state_dict'], self.logger)
      self.optimizer.load_state_dict(state['optimizer'])
      self.scheduler.load_state_dict(state['scheduler'])
      self.resume = True
      if self.is_master:
        self.logger.info("=> loaded checkpoint '{}' (curr_iter {}, curr_epoch {})".format(checkpoint_fn, state['curr_iter'], int(state['curr_iter']//self.n_iter_per_epoch)))
    else:
      if self.is_master: self.logger.info("=> no checkpoint found at '{}'".format(checkpoint_fn))

  def _save_checkpoint(self, curr_iter, filename='checkpoint'):
    if not self.is_master:
        return
    _model = self.model.module if get_world_size() > 1 else self.model
    state = {
        'curr_iter': curr_iter,
        'state_dict': _model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config
    }
    filepath = os.path.join('%s/weights'%(self.config.misc.output_dir), f'{filename}.pth')
    self.logger.info("==> Saving checkpoint: {} ...".format(filepath))
    torch.save(state, filepath)
    # Delete symlink if it exists
    if os.path.exists('%s/weights/weights.pth'%(self.config.misc.output_dir)):
      os.remove('%s/weights/weights.pth'%(self.config.misc.output_dir))
    # Create symlink
    os.system('ln -s %s.pth %s/weights/weights.pth'%(filename, self.config.misc.output_dir))

  def reduce_extracted_feature(self, extracted_feature):
    feature = torch.from_numpy(extracted_feature).cuda().to(self.cur_device)
    dist.all_reduce(feature, op=dist.ReduceOp.SUM)
    return feature.cpu().numpy()

  def tensor_list_to_np_array(self, fea_list):
    fea_new = []
    for i in range(len(fea_list)):
      fea_new.append(fea_list[i].to(self.cur_device))
    feature = torch.cat(fea_new, dim=0)
    return feature.cpu().numpy()

  def instance_cluster_statistics(self, label):
    label_size = label.size
    label_reshape = label.reshape(label_size)
    label_dict = Counter(label_reshape)
    for key in label_dict: label_dict[key] /= label_size
    return label_dict
          
  def pooling_according_label(self, pred_array, plabels_array, pool_flag='Max'):
    proto_fea = []
    for batch_id in range(pred_array.shape[0]):
      plabels = plabels_array[batch_id]
      pred = pred_array[batch_id]
      label_list = list(Counter(plabels).keys())
      for label in label_list:
        idx = np.argwhere(plabels == label)
        idx = idx.reshape(idx.shape[0])
        pred_sel = pred[idx]
        if pool_flag == 'Max': pred_sel = np.max(pred_sel, axis=0)
        else: pred_sel = np.mean(pred_sel, axis=0)
        proto_fea.append(np.concatenate((pred_sel,np.array([label])), axis=0))
    return proto_fea

  # Instance feature computation and clustering
  def instance_kmeans_dist(self):
    fea_dim = self.config.net.model_n_out
    num_gpus = self.config.misc.num_gpus
    # initilize multi-gpu fassi for clustering
    index = init_faiss_multi_gpu(dim = fea_dim, num_gpus = num_gpus)
    proto_features_ori, proto_features_rad = [], []
    
    # -------- Instance Feature Computation -------- 
    self.model.eval()
    with torch.no_grad():
      data_loader_iter = self.train_kmean_data_loader.__iter__()
      self.logger.info('Kmeans - Feature Extraction Begin.')
      for curr_iter in range(len(self.train_kmean_data_loader)):
        # point feature extraction from two views
        data = data_loader_iter.next()
        idx = data['idx'].cpu().numpy()
        plabels = data['pseudo_labels'].cpu().numpy()
        x_ori = data['pcd_ori'].cuda(non_blocking=True).to(self.cur_device)
        x_rad = data['pcd_rad'].cuda(non_blocking=True).to(self.cur_device)
        pred_ori = self.model(x_ori).cpu().numpy().astype(np.float32) # B 32 N
        pred_rad = self.model(x_rad).cpu().numpy().astype(np.float32)
        pred_ori = np.ascontiguousarray(pred_ori.transpose(0,2,1), dtype=np.float32) # B N 32
        pred_rad = np.ascontiguousarray(pred_rad.transpose(0,2,1), dtype=np.float32)
        # instance feature computation and recording for each view
        proto_features_ori += self.pooling_according_label(pred_ori, plabels, pool_flag=self.config.data.proto_pool)
        proto_features_rad += self.pooling_according_label(pred_rad, plabels, pool_flag=self.config.data.proto_pool)
        if curr_iter % 20 == 0:
          self.logger.info('Extracted: {}/{}'.format(curr_iter, len(self.train_kmean_data_loader)))
    self.logger.info('Kmeans - Feature Extraction Done.')
    # ------------------------------------------------------
    
    # collect obtained instance features across all GPUs of each view
    current_proto_num = np.array([len(proto_features_ori)])
    proto_num = self.reduce_extracted_feature(current_proto_num)[0]
    self.logger.info('Kmeans - All proto features: {}'.format(proto_num))
    proto_fea_ori = torch.tensor(proto_features_ori).cuda().to(self.cur_device)
    proto_fea_rad = torch.tensor(proto_features_rad).cuda().to(self.cur_device)
    proto_fea_list_ori = self.gather_together(proto_fea_ori)
    proto_fea_list_rad = self.gather_together(proto_fea_rad)
    proto_fea_list_ori = self.tensor_list_to_np_array(proto_fea_list_ori)
    proto_fea_list_rad = self.tensor_list_to_np_array(proto_fea_list_rad)
    self.logger.info('Kmeans - After gather feature shape: {}x{}'.format(proto_fea_list_ori.shape[0], proto_fea_list_ori.shape[1]))
    # split the instance features and corresponding instance indexes
    proto_fea_ori = proto_fea_list_ori[:,:-1]
    proto_fea_rad = proto_fea_list_rad[:,:-1]
    self.logger.info('Kmeans - feature selected dim: {}'.format(proto_fea_ori.shape[1]))
    proto_idx = proto_fea_list_ori[:,-1]
    # mapping instance index from 0 ~ len(all_instance)
    self.idx_proto_pos = dict(zip(list(proto_idx), [i for i in range(proto_idx.shape[0])]))

    # -------- Mini-batch K-means for instance feature clustering of each view -------- 
    self.logger.info('Kmeans - Begin clustering instance in original view.')
    self.centroid_ori, self.cluster_label_ori = get_faiss_centroid(fea_dim, self.instance_cluster_k, self.kmean_n_iter, index, proto_fea_ori, self.logger)
    self.logger.info('Kmeans - Clustering original view done.')
    self.logger.info('Kmeans - Begin clustering instance in transformed view.')
    self.centroid_rad, self.cluster_label_rad = get_faiss_centroid(fea_dim, self.instance_cluster_k, self.kmean_n_iter, index, proto_fea_rad, self.logger)
    self.logger.info('Kmeans - Clustering transformed view done.')
    torch.cuda.empty_cache()
    # -----------------------------------------------------------------------------------

    # make a statistic of the clustering results for each view
    ori_label_info = self.instance_cluster_statistics(self.cluster_label_ori)
    rad_label_info = self.instance_cluster_statistics(self.cluster_label_rad)
    self.logger.info('Kmeans - Original Label Statistic.')
    self.logger.info(ori_label_info)
    self.logger.info('Kmeans - Rotation Label Statistic.')
    self.logger.info(rad_label_info) 

  def init_classifier(self, cluster_k, centroid):
    dim = self.config.net.model_n_out
    classifier = nn.Conv1d(dim, cluster_k, kernel_size=1, stride=1, padding=0, bias=True)
    classifier.weight.data.normal_(0, 0.01)
    classifier.bias.data.zero_()
    classifier.weight.data = torch.from_numpy(centroid).cuda().float().to(self.cur_device).unsqueeze(-1)
    classifier = classifier.cuda()
    for param in classifier.parameters(): param.requires_grad = False 
    return classifier

  # ---- point clustering training --- 
  def train(self):
    start_epoch = 1
    if self.resume: start_epoch = int(self.total_iter // self.n_iter_per_epoch)
    data_loader_iter = self.data_loader.__iter__()
    cluster_opt = False
    for epoch in range(start_epoch, self.config.trainer.total_epochs + 1):
      tic = time.time()
      if epoch > self.begin_clustering_epoch and epoch % self.cluster_update_freq == 0:
        self.instance_kmeans_dist()
        self.classifier_ori = self.init_classifier(self.config.data.cluster_k,self.centroid_ori)
        self.classifier_rad = self.init_classifier(self.config.data.cluster_k,self.centroid_rad)
        cluster_opt = True
      loss = self._train_each_epoch(epoch, data_loader_iter, cluster_opt)
      self.logger.info('Epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))
      if self.is_master:
        # write to tensor board
        self.writer.add_scalar('ins_loss', loss, epoch)
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        # save chechpoint and evaluate
        if epoch % self.config.trainer.save_freq_epoch == 0:
          self._save_checkpoint(self.total_iter, 'ckpt_epoch_{}'.format(epoch))
        else: self._save_checkpoint(self.total_iter, 'current')    

  def _train_each_epoch(self, epoch, data_loader_iter, cluster_opt=False):
    self.cluster_loss_alpha = self.config.trainer.cluster_loss_alpha
    self.logger.info('The cluster loss alpha: {:0.3f}'.format(self.cluster_loss_alpha))
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    dbscan_loss_meter = AverageMeter()
    if cluster_opt: cluster_loss_meter = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for curr_iter in range(len(self.data_loader)):
      # update data time meter
      data = data_loader_iter.next()
      data_time.update(time.time()-end)
      bsz = self.data_loader.batch_size
      dbscan_loss = self._train_iter_point_clustering(data)
      if cluster_opt: cluster_loss = self._train_iter_instance_clustering(data)    
      else: cluster_loss = 0
      batch_loss = dbscan_loss + self.cluster_loss_alpha * cluster_loss
      # update meters
      loss_meter.update(batch_loss, bsz)
      dbscan_loss_meter.update(dbscan_loss, bsz)
      if cluster_opt: cluster_loss_meter.update(cluster_loss, bsz)
      batch_time.update(time.time()-end)
      end = time.time()

      lr = self.scheduler.get_last_lr()[0]
      self.scheduler.step()

      if curr_iter % self.config.trainer.stat_freq == 0 and self.is_master:
        self.writer.add_scalar('train/loss', batch_loss, curr_iter)
        if not cluster_opt:
          self.logger.info("Train: [{:>3d}]/[{:>4d}/{:>4d}] BT={:>0.3f}/{:>0.3f} DT={:>0.3f} Loss={:>0.3f}/{:>0.3f} DBSCAN_Loss={:>0.3f}/{:>0.3f} Lr={:>0.5f}".format(epoch, curr_iter, len(self.data_loader), batch_time.val, batch_time.avg, data_time.avg, batch_loss, loss_meter.avg, dbscan_loss, dbscan_loss_meter.avg, self.scheduler.get_last_lr()[0]))
        else:
          self.logger.info("Train: [{:>3d}]/[{:>4d}/{:>4d}] BT={:>0.3f}/{:>0.3f} DT={:>0.3f} Loss={:>0.3f}/{:>0.3f} DBSCAN_Loss={:>0.3f}/{:>0.3f} Cluster_Loss={:>0.3f}/{:>0.3f} Lr={:>0.5f}".format(epoch, curr_iter, len(self.data_loader), batch_time.val, batch_time.avg, data_time.avg, batch_loss, loss_meter.avg, dbscan_loss, dbscan_loss_meter.avg, cluster_loss, cluster_loss_meter.avg, self.scheduler.get_last_lr()[0]))
      
      self.total_iter += 1

    return loss_meter.avg

  def _train_iter_point_clustering(self, input_dict):
    self.model.train()
    self.optimizer.zero_grad()
    batch_loss = 0
    
    x_input_ori = input_dict['pcd_ori'].cuda()
    x_input_rad = input_dict['pcd_rad'].cuda()
    plabels = input_dict['pseudo_labels'].cpu().numpy()

    loss = self.obtain_point_classifier_loss(x_input_ori, x_input_ori, plabels, pool_flag=self.config.data.proto_pool) + \
           self.obtain_point_classifier_loss(x_input_rad, x_input_rad, plabels, pool_flag=self.config.data.proto_pool) + \
           self.obtain_point_classifier_loss(x_input_ori, x_input_rad, plabels, pool_flag=self.config.data.proto_pool) + \
           self.obtain_point_classifier_loss(x_input_rad, x_input_ori, plabels, pool_flag=self.config.data.proto_pool)
    loss.backward()

    result = {"loss": loss}
    if self.config.misc.num_gpus > 1:
      result = scaled_all_reduce_dict(result, self.config.misc.num_gpus)
    batch_loss += result["loss"].item()

    self.optimizer.step()
    torch.cuda.empty_cache()

    return batch_loss

  def _train_iter_instance_clustering(self, input_dict):
    self.model.train()
    self.optimizer.zero_grad()
    batch_loss = 0
    
    x_input_ori = input_dict['pcd_ori'].cuda().to(self.cur_device)
    x_input_rad = input_dict['pcd_rad'].cuda().to(self.cur_device)
    idx = input_dict['idx'].cpu().numpy()
    plabels = input_dict['pseudo_labels'].cpu().numpy()


    f_ori = self.model(x_input_ori)
    f_rad = self.model(x_input_rad)
    loss = self.obtain_instance_classifier_loss(f_ori, self.cluster_label_ori, self.classifier_ori, plabels) + \
           self.obtain_instance_classifier_loss(f_rad, self.cluster_label_rad, self.classifier_rad, plabels) + \
           self.obtain_instance_classifier_loss(f_ori, self.cluster_label_rad, self.classifier_rad, plabels) + \
           self.obtain_instance_classifier_loss(f_rad, self.cluster_label_ori, self.classifier_ori, plabels)
    loss.backward()

    result = {"loss": loss}
    if self.config.misc.num_gpus > 1:
      result = scaled_all_reduce_dict(result, self.config.misc.num_gpus)
    batch_loss += result["loss"].item()

    self.optimizer.step()
    torch.cuda.empty_cache()
    return batch_loss

  def obtain_point_classifier_loss(self, x, x_ref, plabels_array, pool_flag='Max'):
    pred_array = self.model(x_ref).detach().cpu().numpy().astype(np.float32) 
    feature = self.model(x)

    pred_array = np.ascontiguousarray(pred_array.transpose(0,2,1), dtype=np.float32)
    classifier_loss = 0
    for batch_id in range(pred_array.shape[0]):
      proto_fea, clabel = [], np.zeros((1, pred_array.shape[1]))
      plabels = plabels_array[batch_id]
      pred = pred_array[batch_id]
      single_feature = feature[batch_id].unsqueeze(0)
      label_list = list(Counter(plabels).keys())
      label_index = 0
      proto_fea = np.zeros((len(label_list), self.config.net.model_n_out))
      for label in label_list:
        idx = np.argwhere(plabels == label)
        idx = idx.reshape(idx.shape[0])
        pred_sel = pred[idx]
        if pool_flag == 'Max': pred_sel = np.max(pred_sel, axis=0)
        else: pred_sel = np.mean(pred_sel, axis=0)
        clabel[0][idx] = label_index
        proto_fea[label_index] = pred_sel
        label_index += 1
      single_classifer = self.init_single_classifier(cluster_k=label_index, centroid=proto_fea)
      output = single_classifer(single_feature).to(self.cur_device)
      clabel = torch.from_numpy(clabel).cuda().long().to(self.cur_device)
      classifier_loss += self.criterion(output, clabel)
    return classifier_loss

  def obtain_instance_classifier_loss(self, feature, cluster_label, classifer, plabels_array):
    output = classifer(feature).to(self.cur_device)
    cluster_pos = []
    for batch_id in range(plabels_array.shape[0]):
      plabels = plabels_array[batch_id]
      proto_pos = [self.idx_proto_pos[plabels[i]] for i in range(plabels.shape[0])]
      cluster_pos.append(proto_pos)
    cluster_pos = np.array(cluster_pos)
    label = torch.from_numpy(cluster_label[cluster_pos]).cuda().long().to(self.cur_device)
    loss = self.criterion(output, label)
    return loss
