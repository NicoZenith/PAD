from __future__ import print_function
import argparse
import os
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import *
import random
from network import *
from scipy import stats, optimize, interpolate


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet | mnist')
parser.add_argument('--dataroot', default='./datasets/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--niterC', type=int, default=20, help='number of epochs to train the classifier')
parser.add_argument('--nf', type=int, default=64, help='filters factor')
parser.add_argument('--drop', type=float, default=0.0, help='probably of drop out')
parser.add_argument('--lrC', type=float, default=0.2, help='learning rate of the classifier, default=0.0002')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='baseline', help='folder to output images and model checkpoints')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=str, default='0', help='The ID of the specified GPU')
parser.add_argument("-n", "--n_samples", dest="n_samples", default=500, type=int,  help="Number of samples")
parser.add_argument("-split", "--split", dest="split", default=20, type=int,  help="split")

opt, unknown = parser.parse_known_args()
print(opt)

# specify the gpu id if using only 1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

dir_files = './results/'+opt.dataset+'/'+opt.outf
dir_checkpoint = './checkpoints/'+opt.dataset+'/'+opt.outf

try:
    os.makedirs(dir_files)
except OSError:
    pass




# TSNE setup
n_samples = opt.n_samples
split = opt.split
n_c = opt.num_classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize, is_train=True)
test_dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize, is_train=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=n_samples//split, shuffle=True, num_workers=int(opt.workers), drop_last=True)
train_dataloader2 = torch.utils.data.DataLoader(train_dataset, batch_size=n_samples//split, shuffle=True, num_workers=int(opt.workers), drop_last=True)
train_dataloader3 = torch.utils.data.DataLoader(train_dataset, batch_size=n_samples//split, shuffle=True, num_workers=int(opt.workers), drop_last=True)
train_dataloader4 = torch.utils.data.DataLoader(train_dataset, batch_size=n_samples//split, shuffle=True, num_workers=int(opt.workers), drop_last=True)

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
num_classes = int(opt.num_classes)
batch_size = opt.batchSize

netG = Generator(ngpu, nz=nz, ngf=opt.nf, img_channels=img_channels)
netG.apply(weights_init)
netD = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels)
netD.apply(weights_init)
# send to GPU
netD.to(device)
netG.to(device)

netGe = Generator(ngpu, nz=nz, ngf=opt.nf, img_channels=img_channels)
netGe.apply(weights_init)
netDe = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels)
netDe.apply(weights_init)
# send to GPU
netDe.to(device)
netGe.to(device)


## FID network
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
net_inception = InceptionV3([block_idx])
net_inception = net_inception.to(device)

if os.path.exists(dir_checkpoint+'/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location='cpu')
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')
    
    
if os.path.exists(dir_checkpoint+'/trained3.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained3.pth', map_location='cpu')
    netGe.load_state_dict(checkpoint['generator'])
    netDe.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')


#imgs, _ = next(iter(train_dataloader))
#imgs = imgs.to(device)
#
#imgs2, _ = next(iter(train_dataloader2))
#imgs2 = imgs2.to(device)

all_inception_real = np.zeros((n_samples, 2048))
all_inception_fake = np.zeros((n_samples, 2048))
with torch.no_grad():
    for i in range(split):
        imgs, _ = next(iter(train_dataloader))
        imgs = imgs.to(device)
        imgs2, _ = next(iter(train_dataloader2))
        imgs2 = imgs2.to(device)
        latent_output, _ = netDe(imgs2)
        reconstructed_imgs2 = netGe(latent_output)
        all_inception_real[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(imgs, net_inception)
        all_inception_fake[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(reconstructed_imgs2, net_inception)
    frechet_dist_NREM_early = calculate_frechet(all_inception_real, all_inception_fake, net_inception)
    print("FID NREM early : "+str(frechet_dist_NREM_early))
#

all_inception_real = np.zeros((n_samples, 2048))
all_inception_fake = np.zeros((n_samples, 2048))
with torch.no_grad():
    for i in range(split):
        imgs, _ = next(iter(train_dataloader))
        imgs = imgs.to(device)
        imgs3, _ = next(iter(train_dataloader3))
        imgs3 = imgs3.to(device)
        imgs4, _ = next(iter(train_dataloader4))
        imgs4 = imgs4.to(device)
        latent_output3, _ = netDe(imgs3)
        latent_output4, _ = netDe(imgs4)
        noise = torch.randn(latent_output3.size(), device=device)
        latent_output_dream = 0.25*latent_output3 + 0.25*latent_output4 + 0.5*noise
        rem_imgs = netGe(latent_output_dream)
        all_inception_real[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(imgs, net_inception)
        all_inception_fake[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(rem_imgs, net_inception)
    frechet_dist_REM_early = calculate_frechet(all_inception_real, all_inception_fake, net_inception)
    print("FID REM early : "+str(frechet_dist_REM_early))



all_inception_real = np.zeros((n_samples, 2048))
all_inception_fake = np.zeros((n_samples, 2048))
with torch.no_grad():
    for i in range(split):
        imgs, _ = next(iter(train_dataloader))
        imgs = imgs.to(device)
        imgs2, _ = next(iter(train_dataloader2))
        imgs2 = imgs2.to(device)
        latent_output, _ = netD(imgs2)
        reconstructed_imgs2 = netG(latent_output)
        all_inception_real[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(imgs, net_inception)
        all_inception_fake[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(reconstructed_imgs2, net_inception)
    frechet_dist_NREM_late = calculate_frechet(all_inception_real, all_inception_fake, net_inception)
    print("FID NREM late : "+str(frechet_dist_NREM_late))

all_inception_real = np.zeros((n_samples, 2048))
all_inception_fake = np.zeros((n_samples, 2048))
with torch.no_grad():
    for i in range(split):
        imgs, _ = next(iter(train_dataloader))
        imgs = imgs.to(device)
        imgs3, _ = next(iter(train_dataloader3))
        imgs3 = imgs3.to(device)
        imgs4, _ = next(iter(train_dataloader4))
        imgs4 = imgs4.to(device)
        latent_output3, _ = netD(imgs3)
        latent_output4, _ = netD(imgs4)
        noise = torch.randn(latent_output3.size(), device=device)
        latent_output_dream = 0.25*latent_output3 + 0.25*latent_output4 + 0.5*noise
        rem_imgs = netG(latent_output_dream)
        all_inception_real[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(imgs, net_inception)
        all_inception_fake[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(rem_imgs, net_inception)
    frechet_dist_REM_late = calculate_frechet(all_inception_real, all_inception_fake, net_inception)
    print("FID REM late : "+str(frechet_dist_REM_late))


frechet_dist_NREM = [frechet_dist_NREM_early, frechet_dist_NREM_late]
frechet_dist_REM = [frechet_dist_REM_early, frechet_dist_REM_late]

torch.save({
        'frechet_dist_NREM': frechet_dist_NREM,
        'frechet_dist_REM': frechet_dist_REM,
    }, dir_files+'/frechet_dist.pth')
print(f'Distances successfully saved.')
