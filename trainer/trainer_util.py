import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Function
import numpy as np


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_master_proc(num_gpus):
    """Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints.
    In the multi GPU setting, we assign the master role to the rank 0 process.
    When training using a single GPU, there is only one training processes
    which is considered the master processes.
    """
    return num_gpus == 1 or torch.distributed.get_rank() == 0


def scaled_all_reduce_dict(res_dict, num_gpus):
    """ Reduce a dictionary of tensors. """
    reductions = []
    for k in res_dict:
        reduction = torch.distributed.all_reduce(res_dict[k], async_op=True)
        reductions.append(reduction)
    for reduction in reductions:
        reduction.wait()
    for k in res_dict:
        res_dict[k] = res_dict[k].clone().mul_(1.0 / num_gpus)
    return res_dict


@torch.no_grad()
def gather_together(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


def _hash(arr, M):
  if isinstance(arr, np.ndarray):
    N, D = arr.shape
  else:
    N, D = len(arr[0]), len(arr)

  hash_vec = np.zeros(N, dtype=np.int64)
  for d in range(D):
    if isinstance(arr, np.ndarray):
      hash_vec += arr[:, d] * M**d
    else:
      hash_vec += arr[d] * M**d
  return hash_vec


def load_state(model, weights, logger, lenient_weight_loading=False, strict=True):
  if get_world_size() > 1:
      _model = model.module
  else:
      _model = model  

  if lenient_weight_loading:
    model_state = _model.state_dict()
    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logger.info("Load weights:" + ', '.join(filtered_weights.keys()))
    weights = model_state
    weights.update(filtered_weights)

  _model.load_state_dict(weights, strict=strict)


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0.0
    self.sq_sum = 0.0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    self.sq_sum += val**2 * n
    self.var = self.sq_sum / self.count - self.avg ** 2
