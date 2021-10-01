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
import json
import torchvision.transforms as transforms
import pickle
from PIL import Image
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
test_dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize, is_train=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=n_samples, shuffle=True, num_workers=int(opt.workers), drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=n_samples, shuffle=False, num_workers=int(opt.workers), drop_last=True)

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



if os.path.exists(dir_checkpoint+'/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location='cpu')
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')


classifier = OutputClassifier(nz, num_classes=num_classes)
classifier.to(device)
optimizerC = optim.SGD(classifier.parameters(), lr=opt.lrC)

if os.path.exists(dir_checkpoint + '/trained_classifier.pth'):
    # Load data from last checkpoint
    print('Loading trained classifier...')
    checkpoint = torch.load(dir_checkpoint + '/trained_classifier.pth', map_location='cpu')
    classifier.load_state_dict(checkpoint['classifier'])
    print('Use trained classifier...')
else:
    print('No trained classifier detected, restart training...')

file_name = ['img_latent_tSNE%i.png', 'img_pixel_tSNE%i.png']

for i in range(2):
 
    # Load TSNE
    if (perplexity < 0):
        tsne = TSNE(n_components=2, verbose=1, init='random', random_state=0)
        fig_title = "PCA Initialization"
        figname = os.path.join(dir_files, 'tsne-pca.png')
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000)
        fig_title = "Perplexity = $%d$"%perplexity
        figname = os.path.join(dir_files, file_name[i]%perplexity)


     # Get full batch for encoding
    imgs, labels = next(iter(train_dataloader))
    imgs = imgs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        latent_output, _ = netD(imgs)
        classes = classifier(latent_output)
        print(classes.shape)
    imgs_flat = imgs.view(imgs.size(0), -1)
    # Cluster with TSNE
    if i==0:
        tsne_enc = tsne.fit_transform(latent_output.cpu())
    else:
        tsne_enc = tsne.fit_transform(imgs_flat.cpu())
    tsne = tsne_enc[:n_samples]
    # Convert to numpy for indexing purposes
    labels = labels.cpu().data.numpy()
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    width = 1000
    height = 750
    max_dim = 32
    full_image = Image.new('RGBA', (width, height))
    imgs = unorm(imgs.cpu())
    pil_imgs = []
    for img in imgs:
        #print(img.shape)
        pil_img = transforms.ToPILImage()(img).convert("RGB")
        pil_imgs.append(pil_img)
    #pil_img.save(figname)
    for img, x, y in zip(pil_imgs, tx, ty):
        tile = img
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
    #fig = plt.figure(figsize = (16,12))
    #fig.imshow(full_image)
    full_image.save(figname)

