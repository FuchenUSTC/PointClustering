# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import random
import torch
import transforms3d
import scipy
import scipy.ndimage
import scipy.interpolate
from util import util
from .data_util import angle_axis

# --- For sparse point cloud
class Compose:
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, coords, feats):
    for transform in self.transforms:
      coords, feats = transform(coords, feats)
    return coords, feats


class Jitter:
  def __init__(self, mu=0, sigma=0.01):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats += np.random.normal(self.mu, self.sigma, (feats.shape[0], feats.shape[1]))
    return coords, feats


class ComposeTransS3DIS(object):
  """Composes several transforms together."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, *args):
    for t in self.transforms:
      args = t(*args)
    return args


class ElasticDistortion:
  def __init__(self, distortion_params):
    self.distortion_params = distortion_params

  def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
      noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                   (noise_dim - 2), noise_dim)
    ]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords, feats, labels

  def __call__(self, coords, feats, labels):
    if self.distortion_params is not None:
      if random.random() < 0.95:
        for granularity, magnitude in self.distortion_params:
          coords, feats, labels = self.elastic_distortion(coords, feats, labels, granularity,
                                                          magnitude)
    return coords, feats, labels


class RandomDropout(object):

  def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.dropout_ratio = dropout_ratio
    self.dropout_application_ratio = dropout_application_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < self.dropout_ratio:
      N = len(coords)
      inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
      return coords[inds], feats[inds], labels[inds]
    return coords, feats, labels


class RandomHorizontalFlip(object):

  def __init__(self, upright_axis, is_temporal):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      for curr_ax in self.horz_axes:
        if random.random() < 0.5:
          coord_max = np.max(coords[:, curr_ax])
          coords[:, curr_ax] = coord_max - coords[:, curr_ax]
    return coords, feats, labels


class ChromaticTranslation(object):
  """Add random color to the image, input must be an array in [0,255] or a PIL image"""

  def __init__(self, trans_range_ratio=1e-1):
    """
    trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
    """
    self.trans_range_ratio = trans_range_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
      feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
    return coords, feats, labels


class ChromaticAutoContrast(object):

  def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
    self.randomize_blend_factor = randomize_blend_factor
    self.blend_factor = blend_factor

  def __call__(self, coords, feats, labels):
    if random.random() < 0.2:
      # mean = np.mean(feats, 0, keepdims=True)
      # std = np.std(feats, 0, keepdims=True)
      # lo = mean - std
      # hi = mean + std
      lo = feats[:, :3].min(0, keepdims=True)
      hi = feats[:, :3].max(0, keepdims=True)
      assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

      scale = 255 / (hi - lo)

      contrast_feats = (feats[:, :3] - lo) * scale

      blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
      feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
    return coords, feats, labels


class ChromaticJitter(object):

  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      noise = np.random.randn(feats.shape[0], 3)
      noise *= self.std * 255
      feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
    return coords, feats, labels


# ---- 

# For dense point cloud (convert to torch)
class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std = std
        self.clip = clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
        symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
        xyz1 *= symmetries
        xyz2 = np.clip(np.random.normal(scale=self.std, size=[pc.shape[0], 3]), a_min=-self.clip, a_max=self.clip)
        pc[:, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(
            xyz2).float()

        return pc


class PointcloudScale(object):
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low, self.scale_high = scale_low, scale_high

    def __call__(self, points):
        scaler = np.random.uniform(self.scale_low, self.scale_high, size=[3])
        scaler = torch.from_numpy(scaler).float()
        points[:, 0:3] *= scaler
        return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, points):
        angles_ = self._get_angles()
        Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points

# batch transform
class BatchPointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std, self.clip = std, clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        bsize = pc.size()[0]
        npoint = pc.size()[1]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
            symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
            xyz1 *= symmetries
            xyz2 = np.clip(np.random.normal(scale=self.std, size=[npoint, 3]), a_max=self.clip, a_min=-self.clip)

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(pc.device)) + torch.from_numpy(
                xyz2).float().to(pc.device)

        return pc


class BatchPointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, pc):
        bsize = pc.size()[0]
        normals = pc.size()[2] > 3
        for i in range(bsize):
            angles_ = self._get_angles()
            Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
            Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
            Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

            rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx).to(pc.device)

            if not normals:
                pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix.t())
            else:
                pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix.t())
                pc[i, :, 3:] = torch.matmul(pc[i, :, 3:], rotation_matrix.t())
        return pc


# For point cloud completion (in numpy)
class Compose3D(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data
