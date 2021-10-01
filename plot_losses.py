from __future__ import print_function
import argparse
import os
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import *
from network import *
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet | mnist')
parser.add_argument('--dataroot', default='./datasets/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--is_continue', type=int, default=1, help='Use pre-trained model')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=55, help='number of epochs to train for')
parser.add_argument('--mu', type=float, default=1.0, help='weight of Cycle cWonsistency')
parser.add_argument('--W', type=float, default=1.0, help='Wake rec weight')
parser.add_argument('--N', type=float, default=1.0, help='NREM sleep weight')
parser.add_argument('--R', type=float, default=1.0, help='PGO REM sleep (GANs)')
parser.add_argument('--epsilon', type=float, default=0.0, help='amount of noise in wake latent space')
parser.add_argument('--nf', type=int, default=64, help='filters factor')
parser.add_argument('--drop', type=float, default=0.0, help='probably of drop out')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lmbd', type=float, default=0.5, help='convex combination factor for REM')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='eval_images', help='folder to output images and model checkpoints')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
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


d_losses = []
g_losses = []
r_losses_real = []
r_losses_fake = []
kl_losses = []

if os.path.exists(dir_checkpoint+'/trained.pth') and opt.is_continue:
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location=torch.device('cpu'))
    d_losses = checkpoint.get('d_losses', [float('inf')])
    g_losses = checkpoint.get('g_losses', [float('inf')])
    r_losses_real = checkpoint.get('r_losses_real', [float('inf')])
    r_losses_fake = checkpoint.get('r_losses_fake', [float('inf')])
    kl_losses = checkpoint.get('kl_losses', [float('inf')])
    print('Losses found...')
else:
    print('No loss found...')
    
epoch = len(d_losses)-1

e = np.arange(0, epoch+1)
fig = plt.figure(figsize=(5,4))
ax1 = fig.add_subplot(111)
if r_losses_real is not None:
    ax1.plot(e, r_losses_real, color='orange', label='$\mathcal{L}_{\mathrm{img}}$')
if kl_losses is not None:
    ax1.plot(e, np.array(kl_losses) -0.5, color='brown', label='$\mathcal{L}_{\mathrm{KL}}$')
if r_losses_fake is not None:
    ax1.plot(e, r_losses_fake, color='magenta', label='$\mathcal{L}_{\mathrm{latent}}$')
if d_losses is not None:
    ax1.plot(e, d_losses, color='green', label=' $\mathcal{L}_{\mathrm{real}}$ + $\mathcal{L}_{\mathrm{fake}}$')
if g_losses is not None:
    ax1.plot(e, g_losses, label='- $\mathcal{L}_{\mathrm{fake}}$')
#ax1.set_ylim(0, 10)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
#ax1.set_title('losses with training')



ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
for axis in 'left', 'bottom':
  ax1.spines[axis].set_linewidth(1.5)
ax1.set_ylim(-1, 1.5)
ax1.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
plt.tight_layout()
plt.tight_layout()

ax1.legend(loc="best", frameon=True, fontsize=12)

fig.savefig(dir_files+'/losses.pdf')
