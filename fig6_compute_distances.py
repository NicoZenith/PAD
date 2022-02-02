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
from scipy import stats, optimize, interpolate
from sklearn.decomposition import PCA


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

opt, unknown = parser.parse_known_args()
print(opt)

# specify the gpu id if using only 1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

##dir_files = './results/'+opt.dataset+'/'+opt.outf
#dir_checkpoint = './checkpoints/'+opt.dataset+'/'+opt.outf
dataset = 'svhn'
folder = './results/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
idx = ['', '1', '2', '3']
dir_files_0, dir_files_1, dir_files_2, dir_files_3 = [],[],[],[]
for i in idx:
    dir_files_0.append('./checkpoints/' + dataset + '/nomodel' + i)
    dir_files_1.append('./checkpoints/' + dataset + '/model_mix_wnr'+i)
    dir_files_2.append('./checkpoints/' + dataset + '/model_wn' + i)
    dir_files_3.append('./checkpoints/' + dataset + '/model_mix_wr' + i)
    
all_dir_files = [dir_files_0, dir_files_1, dir_files_2,  dir_files_3]

# for ratios, 0-3 are not for files (different net init), but for conditions: no training, full training, no REM, no NREM
ratio0 = np.zeros((len(dir_files_1)))
ratio1 = np.zeros((len(dir_files_1)))
ratio2 = np.zeros((len(dir_files_1)))
ratio3 = np.zeros((len(dir_files_1)))

all_ratios = [ratio0, ratio1, ratio2, ratio3]

distance0 = np.zeros((len(dir_files_1)))
distance1 = np.zeros((len(dir_files_1)))
distance2 = np.zeros((len(dir_files_1)))
distance3 = np.zeros((len(dir_files_1)))

all_distances = [distance0, distance1, distance2, distance3]

def get_indices(dataset,dataset_name, class_name):
    indices =  []
    if dataset_name == 'cifar10':
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == class_name:
                indices.append(i)
    else:
        for i in range(len(dataset.labels)):
            if dataset.labels[i] == class_name:
                indices.append(i)
    return indices
    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset, unorm, img_channels = get_dataset(dataset, opt.dataroot, opt.imageSize, is_train=True)
test_dataset, unorm, img_channels = get_dataset(dataset, opt.dataroot, opt.imageSize, is_train=False)
if dataset =='cifar10':
    all_indices = [k for k in range(50000)]
else:
    all_indices = [k for k in range(73257)]


indices_per_class = [get_indices(train_dataset,dataset, i) for i in range(opt.num_classes)]
indicesA = [idx[:len(idx)//2] for idx in indices_per_class]
indicesB = [idx[len(idx)//2:] for idx in indices_per_class]
dataloadersA = [torch.utils.data.DataLoader(train_dataset,batch_size=opt.n_samples, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)) for idx in indicesA]
dataloadersB = [ torch.utils.data.DataLoader(train_dataset,batch_size=opt.n_samples, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)) for idx in indicesB]
# for interclass, make sure that we pick all but images of the same class as dataloadersA
indicesC = [[k for k in all_indices if k not in indices_per_class[i]] for i in range(opt.num_classes)] #takes time
dataloadersC = [ torch.utils.data.DataLoader(train_dataset,batch_size=opt.n_samples, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)) for idx in indicesC]


# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
batch_size = opt.batchSize

netD = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels)
netD.apply(weights_init)
netD.to(device)

rec_criterion = nn.MSELoss() # reconstruction


for k in range(len(dir_files_1)):
    
    for condition in range(len(all_dir_files)):
        
        ## check if the model exists. If not, random init (netD0). If
        if  os.path.exists(all_dir_files[condition][k]+'/trained.pth'):
            # Load data from last checkpoint
            print('Loading pre-trained model...')
            checkpoint = torch.load(all_dir_files[condition][k]+'/trained.pth', map_location='cpu')
            netD.load_state_dict(checkpoint['discriminator'])
            print('Using loaded model...')
        else:
            # Load data from last checkpoint
            netD = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels)
            netD.apply(weights_init)
            netD.to(device)
            print('No pre-trained model detected, use randomly initialized model...')

        

        intraclass_distance = 0
        interclass_distance = 0
        clean_occluded_distance = 0
        for i in range(len(indicesA)):
            imageA, _ = next(iter(dataloadersA[i]))
            occlusion = Occlude(drop_rate=0.3, tile_size=4)
            imageA_occ = occlusion(imageA, d=1)
            imageB, _ = next(iter(dataloadersB[i]))
            imageC, _ = next(iter(dataloadersC[i]))

            
            with torch.no_grad():
                latent_outputsA, _ = netD(imageA)
                latent_outputsA_occ, _ = netD(imageA_occ)
                latent_outputsB, _ = netD(imageB)
                latent_outputsC, _ = netD(imageC)
                #latent_outputs = torch.tensor(pca.fit_transform(torch.cat((latent_outputsA, latent_outputsB), dim=0)))
            intraclass_distance += rec_criterion(latent_outputsA, latent_outputsB).item()
            interclass_distance += rec_criterion(latent_outputsA, latent_outputsC).item()
            clean_occluded_distance += rec_criterion(latent_outputsA, latent_outputsA_occ).item()
        intraclass_distance = intraclass_distance/len(indicesA)
#        print(intraclass_distance)
        interclass_distance = interclass_distance/len(indicesA)
#        print(interclass_distance)
        clean_occluded_distance = clean_occluded_distance/len(indicesA)
        all_ratios[condition][k] = intraclass_distance/interclass_distance
        print(all_ratios[condition][k])
        all_distances[condition][k] = clean_occluded_distance/interclass_distance
        print(all_distances[condition][k])
    
# do checkpointing

torch.save({
    'all_ratios': all_ratios,
    'all_distances': all_distances,
}, folder+dataset+'_clustering_metrics.pth')
print(f'Distances successfully saved.')






