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
from sklearn.manifold import TSNE


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
parser.add_argument("-p", "--perplexity", dest="perplexity", default=30, type=int,  help="TSNE perplexity")
parser.add_argument("-n", "--n_samples", dest="n_samples", default=500, type=int,  help="Number of samples")

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
perplexity = opt.perplexity
n_c = opt.num_classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize, is_train=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=n_samples, shuffle=True, num_workers=int(opt.workers), drop_last=True)

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
num_classes = int(opt.num_classes)
batch_size = opt.batchSize

netD = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels)
netD.apply(weights_init)
# send to GPU
netD.to(device)



if os.path.exists(dir_checkpoint+'/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location='cpu')
    netD.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, randomly init network...')


 
 
# Load TSNE
if (perplexity < 0):
    tsne = TSNE(n_components=2, verbose=1, init='random', random_state=0)
    fig_title = "PCA Initialization"
    figname = os.path.join(dir_files, 'tsne-pca.png')
else:
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000)
    fig_title = "Perplexity = $%d$"%perplexity
    figname = os.path.join(dir_files, 'tsne-plex%i.png'%perplexity)


 # Get full batch for encoding
imgs, labels = next(iter(train_dataloader))
imgs = imgs.to(device)
occlusion = Occlude(drop_rate=0.3, tile_size=4)
imgs_occ = occlusion(imgs, d=1)
labels = labels.to(device)
with torch.no_grad():
    latent_output, _ = netD(imgs)
    latent_output_occ, _ = netD(imgs_occ)
# Cluster with TSNE
tsne_enc = tsne.fit_transform(torch.cat((latent_output, latent_output_occ), dim=0).cpu())
tsne = tsne_enc[:n_samples]
tsne_occ = tsne_enc[n_samples:]
# Convert to numpy for indexing purposes
labels = labels.cpu().data.numpy()

# Color and marker for each true class
colors = cm.rainbow(np.linspace(0, 1, 4))
markers = matplotlib.markers.MarkerStyle.filled_markers
n_points = 50
list_cifar = ['plane','car','bird','cat']
# Save TSNE figure to file
fig, ax = plt.subplots(figsize=(16, 10))
# for clean inputs
for iclass in range(0, 4):
    # Get indices for each class
    idxs = labels == iclass
    # Scatter those points in tsne dims
    if opt.dataset=='cifar10':
        label = list_cifar[iclass]
    else:
        label = r'$%i$' % iclass
    ax.scatter(tsne[idxs, 0][:n_points],
               tsne[idxs, 1][:n_points],
               marker='o',
               c=colors[iclass],
               edgecolor=None,
               linewidth=1,
               s=200,
               label=label)
               
# for occluded inputs
for iclass in range(0, 4):
    # Get indices for each class
    idxs = labels == iclass
    # Scatter those points in tsne dims
    ax.scatter(tsne_occ[idxs, 0][:n_points],
               tsne_occ[idxs, 1][:n_points],
               marker='o',
               color = 'none',
               edgecolors=colors[iclass],
               linewidth=2,
               s=600,
               label=None)

ax.axis('off')

fig.savefig(figname)
