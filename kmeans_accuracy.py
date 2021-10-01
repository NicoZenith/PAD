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
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
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

folder = './results/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
idx = ['', '1', '2', '3']
n_samples = opt.n_samples
dir_files_1, dir_files_2, dir_files_3, dir_files_4 = [],[],[],[]
for i in idx:
    dir_files_1.append('./checkpoints/'+opt.dataset+'/model_wnr'+i)
    dir_files_2.append('./checkpoints/' + opt.dataset + '/model_wn' + i)
    dir_files_3.append('./checkpoints/' + opt.dataset + '/model_wr' + i)
    dir_files_4.append('./checkpoints/' + opt.dataset + '/model_wr' + i)

accuracies1 = np.zeros((len(dir_files_1)))
accuracies2 = np.zeros((len(dir_files_1)))
accuracies3 = np.zeros((len(dir_files_1)))
accuracies4 = np.zeros((len(dir_files_1)))



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

train_dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize, is_train=True)
test_dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize, is_train=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=n_samples, shuffle=True, num_workers=int(opt.workers), drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=n_samples, shuffle=False, num_workers=int(opt.workers), drop_last=True)

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
batch_size = opt.batchSize


netD1 = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels)
netD1.apply(weights_init)
netD1.to(device)
classifier1 = OutputClassifier(nz, num_classes=opt.num_classes)
classifier1.to(device)

netD2 = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels)
netD2.apply(weights_init)
netD2.to(device)
classifier2 = OutputClassifier(nz, num_classes=opt.num_classes)
classifier2.to(device)

netD3 = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels)
netD3.apply(weights_init)
netD3.to(device)
classifier3 = OutputClassifier(nz, num_classes=opt.num_classes)
classifier3.to(device)



def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)


# clustering
km = KMeans(n_clusters=10, random_state=0)
pca = PCA(whiten=True)


for k in range(len(dir_files_1)):

    ## FULL MODEL
    if os.path.exists(dir_files_1[k]+'/trained.pth'):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load(dir_files_1[k]+'/trained.pth', map_location='cpu')
        netD1.load_state_dict(checkpoint['discriminator'])
        checkpoint = torch.load(dir_files_1[k] + '/trained_classifier.pth', map_location='cpu')
        classifier1.load_state_dict(checkpoint['classifier'])
        print('Using loaded model...')
    else:
        print('No pre-trained model detected, restart training...')
    

    imgs, labels = next(iter(train_dataloader))
    imgs = imgs.to(device)
    with torch.no_grad():
        latent_outputs, _ = netD1(imgs)
        #print(latent_outputs)
        #latent_outputs = pca.fit_transform(latent_outputs)
        #print(latent_outputs)
        km.fit(latent_outputs)
        predicted_labels = km.labels_

    cm = confusion_matrix(labels, predicted_labels)
    #print(cm)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]

    #print(cm2)
    accuracy = np.trace(cm2) / np.sum(cm2)
    accuracies1[k] = accuracy
    print(accuracy)
    
    
    
    
    ## w/o REM
    if os.path.exists(dir_files_2[k]+'/trained.pth'):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load(dir_files_2[k]+'/trained.pth', map_location='cpu')
        netD2.load_state_dict(checkpoint['discriminator'])
        checkpoint = torch.load(dir_files_2[k] + '/trained_classifier.pth', map_location='cpu')
        classifier2.load_state_dict(checkpoint['classifier'])
        print('Using loaded model...')
    else:
        print('No pre-trained model detected, restart training...')

    # Get full batch for encoding
    with torch.no_grad():
        latent_outputs, _ = netD2(imgs)
        #print(latent_outputs)
        #latent_outputs = pca.fit_transform(latent_outputs)
        #print(latent_outputs)
        km.fit(latent_outputs)
        predicted_labels = km.labels_

    cm = confusion_matrix(labels, predicted_labels)
    #print(cm)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]

    #print(cm2)
    accuracy = np.trace(cm2) / np.sum(cm2)
    accuracies2[k] = accuracy
    print(accuracy)

    
    
    ## w/o NREM
    if os.path.exists(dir_files_3[k]+'/trained.pth'):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load(dir_files_3[k]+'/trained.pth', map_location='cpu')
        netD3.load_state_dict(checkpoint['discriminator'])
        checkpoint = torch.load(dir_files_3[k] + '/trained_classifier.pth', map_location='cpu')
        classifier3.load_state_dict(checkpoint['classifier'])
        print('Using loaded model...')
    else:
        print('No pre-trained model detected, restart training...')

    with torch.no_grad():
        latent_outputs, _ = netD3(imgs)
        #print(latent_outputs)
        #latent_outputs = pca.fit_transform(latent_outputs)
        #print(latent_outputs)
        km.fit(latent_outputs)
        predicted_labels = km.labels_
    cm = confusion_matrix(labels, predicted_labels)
    #print(cm)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]

    #print(cm2)
    accuracy = np.trace(cm2) / np.sum(cm2)
    accuracies3[k] = accuracy
    print(accuracy)
    
all_accuracies = [accuracies1, accuracies2, accuracies3]
    


torch.save({
    'all_accuracies': all_accuracies,
}, folder+opt.dataset+'_clustering_accuracies.pth')
print(f'Distances successfully saved.')





