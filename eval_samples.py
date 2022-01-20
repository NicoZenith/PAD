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
parser.add_argument('--batchSize', type=int, default=64, help='input batch   size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--mu', type=float, default=0.0, help='weight of Cycle cWonsistency')
parser.add_argument('--occ', type=float, default=0.0, help='probability of occlusion')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='bnoise_a1a2b1b2', help='folder to output images and model checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='The ID of the specified GPU')

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
try:
    os.makedirs(dir_checkpoint)
except OSError:
    pass

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize, is_train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True)

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
batch_size = opt.batchSize
lmbd = 0.5

# Load final epoch networks 
netG = Generator(ngpu, nz=nz, img_channels=img_channels)
netG.apply(weights_init)
netD = Discriminator(ngpu, nz=nz, img_channels=img_channels,  p_drop=0.0)
netD.apply(weights_init)
# send to GPU
netD.to(device)
netG.to(device)

if os.path.exists(dir_checkpoint+'/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location='cpu')
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')
    
# load epoch 3 networks
netG3 = Generator(ngpu, nz=nz, img_channels=img_channels)
netG3.apply(weights_init)
netD3 = Discriminator(ngpu, nz=nz, img_channels=img_channels,  p_drop=0.0)
netD3.apply(weights_init)
# send to GPU
netD3.to(device)
netG3.to(device)

if os.path.exists(dir_checkpoint+'/trained2.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained2.pth', map_location='cpu')
    netG3.load_state_dict(checkpoint['generator'])
    netD3.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')

# Prepare images
dataloader_iter = iter(dataloader)
image_eval1, _ = next(dataloader_iter) # first batch of images
image_eval2, _ = next(dataloader_iter) # second batch of images
image_eval1_save = image_eval1.clone()
image_eval2_save = image_eval2.clone()
 # keep only the 3 first images of each batch
vutils.save_image(unorm(image_eval1_save[:3]).data, '%s/eval_wake1_%03d.png' % (dir_files, 0), nrow=1)
vutils.save_image(unorm(image_eval2_save[:3]).data, '%s/eval_wake2_%03d.png' % (dir_files, 0), nrow=1)


# generate samples with final epoch networks
with torch.no_grad():
    image_eval1 = image_eval1.to(device)
    image_eval2 = image_eval2.to(device)
    latent_output1, _ = netD(image_eval1)
    latent_output2, _ = netD(image_eval2)
    #print(latent_output1.shape)
    rec_image_eval1 = netG(latent_output1)
    rec_image_eval2 = netG(latent_output2)
    occlusion_eval = Occlude(drop_rate=0.2, tile_size=4) # add occlusions on NREM sample
    nrem = occlusion_eval(rec_image_eval1, d=1)
    noise = torch.randn(batch_size, nz, device=device)
    latent_rem = 0.25*latent_output1 + 0.25*latent_output2 + 0.5*noise
    
    rem = netG(latent_rem)
nrem = unorm(nrem)
rem = unorm(rem)
rec_image_eval1=unorm(rec_image_eval1)
rec_image_eval2=unorm(rec_image_eval2)
vutils.save_image(rec_image_eval1[:3].data, '%s/eval_rec1.png' % (dir_files), nrow=1)
vutils.save_image(rec_image_eval2[:3].data, '%s/eval_rec2.png' % (dir_files), nrow=1)
vutils.save_image(nrem[:3].data, '%s/eval_nrem.png' % (dir_files), nrow=1)
vutils.save_image(rem[:3].data, '%s/eval_rem.png' % (dir_files), nrow=1)

# generate samples with epoch 3 networks
with torch.no_grad():
    image_eval1 = image_eval1.to(device)
    image_eval2 = image_eval2.to(device)
    latent_output1, _ = netD3(image_eval1)
    latent_output2, _ = netD3(image_eval2)
    #print(latent_output1.shape)
    rec_image_eval1 = netG3(latent_output1)
    rec_image_eval2 = netG3(latent_output2)
    occlusion_eval = Occlude(drop_rate=0.2, tile_size=4)
    nrem = occlusion_eval(rec_image_eval1, d=1)
    noise = torch.randn(batch_size, nz, device=device)
    latent_rem = 0.25*latent_output1 + 0.25*latent_output2 + 0.5*noise
    rem = netG3(latent_rem)
nrem = unorm(nrem)
rem = unorm(rem)
rec_image_eval1=unorm(rec_image_eval1)
rec_image_eval2=unorm(rec_image_eval2)
vutils.save_image(rec_image_eval1[:3].data, '%s/eval3_rec1.png' % (dir_files), nrow=1)
vutils.save_image(rec_image_eval2[:3].data, '%s/eval3_rec2.png' % (dir_files), nrow=1)
vutils.save_image(nrem[:3].data, '%s/eval3_nrem.png' % (dir_files), nrow=1)
vutils.save_image(rem[:3].data, '%s/eval3_rem.png' % (dir_files), nrow=1)



