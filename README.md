# [CVPR 2023] PointClustering: Unsupervised Point Cloud Pre-training using Transformation Invariance in Clustering

This repository includes the PointClustering codes and related configurations. 

The downstream tasks training code and configurations will be coming soon.

# Update 
* 2023.6.1: Repository for PointClustering

# Contents:

* [Paper Introduction](#paper-introduction)
* [Required Environment](#required-environment)
* [Training of PointClustering](#training-of-pointclustering)
* [Citation](#citation)

# Paper Introduction
<div align=center>
<img src="https://raw.githubusercontent.com/FuchenUSTC/PointClustering/master/pic/point_clustering.JPG" width="300" alt="image"/>
</div>

Feature invariance under different data transformations, i.e., transformation invariance, can be regarded as a type of self-supervision for representation learning. In this paper, we present PointClustering, a new unsupervised representation learning scheme that leverages transformation invariance for point cloud pre-training. PointClustering formulates the pretext task as deep clustering and employs transformation invariance as an inductive bias, following the philosophy that common point cloud transformation will not change the geometric properties and semantics. Technically, PointClustering iteratively optimizes the feature clusters and backbone, and delves into the transformation invariance as learning regularization from two perspectives: point level and instance level. Point-level invariance learning maintains local geometric properties through gathering point features of one instance across transformations, while instance-level invariance learning further measures clusters over the entire dataset to explore semantics of instances. Our PointClustering is architecture-agnostic and readily applicable to MLP-based, CNN-based and Transformer-based backbones. We empirically demonstrate that the models pre-learnt on the ScanNet dataset by PointClustering provide superior performances on six benchmarks, across downstream tasks of classification and segmentation.

# Required Environment

- python 3.8.0
- Pytorch 1.7
- CUDA 10.1
- cuDNN 8.0
- GPU NVIDIA Tesla P40 (24GB x4)

To guarantee the success of compiling SIFA cuda kernel, the nvcc cuda compiler should be installed in the environment. We have integrated the complete running environment into a docker image and will release it on DockerHub in the future.


# Training of PointClustering

Will be coming soon.


# Citation

If you use these models in your research, please cite:

    @inproceedings{Long:CVPR23,
      title={PointClustering: Unsupervised Point Cloud Pre-training using Transformation Invariance in Clustering},
      author={Fuchen Long, Ting Yao, Zhaofan Qiu, Lusong Li and Tao Mei},
      booktitle={CVPR},
      year={2023}
    }