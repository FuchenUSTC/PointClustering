import torch
import torch.nn.functional as F
import numpy as np
import time
import faiss

from sklearn.datasets import make_blobs


def init_faiss_multi_gpu(dim, num_gpus):
  flat_config = []
  res = [faiss.StandardGpuResources() for i in range(num_gpus)]
  for i in range(num_gpus):
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = i
    flat_config.append(cfg)
  indexes = [faiss.GpuIndexFlatL2(res[i], dim, flat_config[i]) for i in range(num_gpus)]
  index = faiss.IndexReplicas()
  for sub_index in indexes: index.addIndex(sub_index)
  return index


def get_faiss_centroid(dim, cluster_k, kmean_n_iter, index, feature, logger=None):
  index.reset()
  feature = feature.reshape(-1, dim)
  feature = np.ascontiguousarray(feature, dtype=np.float32)
  if logger == None: print('Kmeans - Clustering k is {}'.format(cluster_k))
  else: logger.info('Kmeans - Clustering k is {}'.format(cluster_k))
  clus = faiss.Clustering(dim, cluster_k)
  clus.verbose = False
  clus.seed = 1
  clus.niter = kmean_n_iter
  clus.max_points_per_centroid = 20000000
  clus.train(feature, index)
  centroid = faiss.vector_float_to_array(clus.centroids).reshape(cluster_k, dim)
  D, I = index.search(feature, 1) # for each sample, find cluster distance and assignments
  im2cluster = np.array([int(n[0]) for n in I])
  return centroid, im2cluster

# -- K-means testing --- #

# Generate 2D points around [0,0], [1,1], [-1,1] for clustering testing
def clustering_data_generator(n_samples):
  centers = [[0,0], [1,1], [-1,1]]
  X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.6, random_state=0)
  X = X.astype(np.float32)
  return X


def test_multi_gpu_kmeans():
    """
    Args:
        x: data to be clustered
    """
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    temperature = 0.2    
    x = clustering_data_generator(n_samples=819200)
    num_cluster = 3
    kmean_n_iter = 100
    if torch.distributed.get_rank() == 0: print('performing kmeans clustering')
    dim = x.shape[1]

    # Multi gpu clustering
    index = init_faiss_multi_gpu(dim, 4)
    if torch.distributed.get_rank() == 0: 
      st = time.time()
    centroid, label = get_faiss_centroid(dim, num_cluster, kmean_n_iter, index, x)
    if torch.distributed.get_rank() == 0: 
      run_time = time.time() - st
      print('Multi Gpu clustering run time: %0.2f'%(run_time))
    if torch.distributed.get_rank() == 0: print(centroid)

# ---------------------- #

if __name__ == '__main__':
  print('--------- multi-gpu mini-bach K-means testing ---------')
  test_multi_gpu_kmeans()
  print('--------------------- Done ---------------------')