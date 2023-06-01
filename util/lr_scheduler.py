# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2022/01/06
# desc: learning scheduler and optimizer

# noinspection PyProtectedMember
import torch
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR, LambdaLR
from torch.optim import SGD, Adam, AdamW
from torch import nn


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn


# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


class LambdaStepLR(LambdaLR):

  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v


class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""

  def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)


class SquaredLR(LambdaStepLR):
  """ Used for SGD Lars"""

  def __init__(self, optimizer, max_iter, last_step=-1):
    super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**2, last_step)


class ExpLR(LambdaStepLR):

  def __init__(self, optimizer, step_size, gamma=0.9, last_step=-1):
    # (0.9 ** 21.854) = 0.1, (0.95 ** 44.8906) = 0.1
    # To get 0.1 every N using gamma 0.9, N * log(0.9)/log(0.1) = 0.04575749 N
    # To get 0.1 every N using gamma g, g ** N = 0.1 -> N * log(g) = log(0.1) -> g = np.exp(log(0.1) / N)
    super(ExpLR, self).__init__(optimizer, lambda s: gamma**(s / step_size), last_step)


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


def get_lambda_bnsche(model, config):
    if config.opt.bnmscheduler.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.opt.bnmscheduler.bn_momentum * config.opt.bnmscheduler.bn_decay ** (e / config.opt.bnmscheduler.decay_step), config.opt.bnmscheduler.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler


def get_scheduler(optimizer, config, n_iter_per_epoch, model):
    if config.opt.scheduler == 'Cosine':
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001,
            T_max=(config.trainer.total_epochs - config.trainer.warmup_epoch) * n_iter_per_epoch)
    elif config.opt.scheduler == 'Step':
        lr_decay_epochs = [config.opt.lr_decay_steps * i for i in range(1, config.trainer.total_epochs // config.opt.lr_decay_steps)]
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=config.opt.lr_decay_rate,
            milestones=[(m - config.trainer.warmup_epoch) * n_iter_per_epoch for m in lr_decay_epochs])
    elif config.opt.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.opt.exp_step_size, gamma=config.opt.exp_gamma)
    elif config.opt.scheduler == 'PolyLR':
        scheduler = PolyLR(optimizer, max_iter=config.opt.max_iter, power=config.opt.poly_power)
    elif config.opt.scheduler == 'SquaredLR':
        scheduler = SquaredLR(optimizer, max_iter=config.opt.max_iter)
    elif config.opt.scheduler == 'ExpLR':
        scheduler = ExpLR(optimizer, step_size=config.opt.exp_step_size, gamma=config.opt.exp_gamma)
    else:    
        raise NotImplementedError(f"scheduler {args.lr_scheduler} not supported")

    if config.trainer.warmup_epoch > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=config.opt.warmup_multiplier,
            after_scheduler=scheduler,
            warmup_epoch=config.trainer.warmup_epoch * n_iter_per_epoch)
    
    # scheduler batch_normalization momentum
    if config.opt.get('bnmscheduler') is not None:
        if config.opt.bnmscheduler.type == 'Lambda':
            bnscheduler = get_lambda_bnsche(model, config)
        scheduler = [scheduler, bnscheduler]

    return scheduler


def get_optimizer(params, config):
    if config.opt.optimizer == 'SGD':
        optimizer = SGD(params, lr=config.opt.lr,
                        momentum=config.opt.momentum,
                        weight_decay=config.opt.weight_decay)        
    elif config.opt.optimizer == 'Adam':
        optimizer = Adam(params, lr=config.opt.lr,
                         betas=(config.opt.adam_beta1, config.opt.adam_beta2),
                         eps = config.opt.adam_eps,
                         weight_decay=config.opt.weight_decay)          
    elif config.opt.optimizer == 'AdamW':
        optimizer = AdamW(params, lr=config.opt.lr,
                          betas=(config.opt.adam_beta1, config.opt.adam_beta2),
                          weight_decay=config.opt.weight_decay)    
    else:
        raise ValueError('Optimizer type not supported')
    return optimizer