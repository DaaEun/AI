# A dataset of tiny images
# CIFAR-10 : Canadian Institute For Advanced Research

from matplotlib import pyplot as plt
import numpy as np
import torch

torch.set_printoptions(edgeitems=2, linewidth=75)
torch.manual_seed(123)

##############################
#### Dowmloading CIFAR-10 ####
##############################

from torchvision import datasets
import ssl

# 다운로드한 파일 저장할 경로
data_path = 'data-unversioned/6_datasets/'  
# 한 번 다운로드 후 download=False로 변경
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)   
# train=False : validation set이기 때문에     
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)   