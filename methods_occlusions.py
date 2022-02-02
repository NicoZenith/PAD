from __future__ import print_function
import argparse
import os
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import *
from network import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet | mnist')
parser.add_argument('--dataroot', default='./datasets/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

opt, unknown = parser.parse_known_args()
print(opt)

dir_files = './results/'+opt.dataset
try:
    os.makedirs(dir_files)
except OSError:
    pass

dataset, unorm, img_channels = get_dataset(dataset_name=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize, is_train=True)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

imgs, _ = next(iter(train_dataloader))
imgs = imgs.to(device)
probas = np.arange(0, 1.1, 0.1)
imgs_array = torch.zeros(len(probas), imgs.size(1), imgs.size(2), imgs.size(3))
for i in range(len(probas)):
    occlusion = Occlude(drop_rate=probas[i], tile_size=4)
    imgs_occ = occlusion(imgs, d=1)
    selected_img_occ = imgs_occ[0]
    imgs_array[i] = selected_img_occ

vutils.save_image(unorm(imgs_array), '%s/imgs_array.png' % dir_files, nrow=11)


