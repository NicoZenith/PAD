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

folder = './results/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# do checkpointing
occlusions = True
if occlusions:
    print('Loading distances...')
    all_distances_cifar10 = torch.load(folder+'cifar10'+'_clustering_metrics.pth', map_location=device).get('all_distances', [float('inf')])
    all_distances_svhn = torch.load(folder+'svhn'+'_clustering_metrics.pth', map_location=device).get('all_distances', [float('inf')])
    print('Distances found...')
else:
    print('Loading distances...')
    all_distances_cifar10 = torch.load(folder+'cifar10'+'_clustering_metrics.pth', map_location=device).get('all_ratios', [float('inf')])
    all_distances_svhn = torch.load(folder+'svhn'+'_clustering_metrics.pth', map_location=device).get('all_ratios', [float('inf')])
    print('Distances found...')



    
def mean_and_err(array, axis=0):
    mean = array.mean(axis=0)
    sem = stats.sem(array, axis=axis)
    return mean, sem
    
    

#x = ['Early training','Full model', 'w/o REM', 'w/o NREM']
y_cifar10 = [mean_and_err(all_distances_cifar10[k])[0] for k in range(4)]
y_svhn =  [mean_and_err(all_distances_svhn[k])[0] for k in range(4)]
err_cifar10 = [mean_and_err(all_distances_cifar10[k])[1] for k in range(4)]
print(err_cifar10)
err_svhn = [mean_and_err(all_distances_svhn[k])[1] for k in range(4)]
colors = ['gray', 'black', 'magenta', 'darkorange']
edgecolors = ['black', 'gray', 'black', 'black']


fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
ind = np.arange(4)  # the x locations for the groups
width = 0.35
ax.bar(ind - width/2, y_cifar10, yerr=err_cifar10, linewidth=1, edgecolor=edgecolors, color=colors, width=width, capsize=5)
ax.bar(ind + width/2, y_svhn, yerr=err_svhn, linewidth=1, edgecolor=edgecolors, color=colors, width=width, capsize=5)

if occlusions:
    ax.set_ylabel("clean-occluded/inter-class ratio", fontsize=14, labelpad=5)
else:
    ax.set_ylabel("intra/inter-class ratio", fontsize=14, labelpad=5)

ax.set_xticklabels(('No\ntraining','Full\nmodel', 'w/o\nREM', 'w/o\nNREM'),  fontsize=14)
ax.set_xticks(ind)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for axis in 'left', 'bottom':
  ax.spines[axis].set_linewidth(1.5)
ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
plt.tight_layout()
plt.tight_layout()
if not occlusions:
    ax.set_ylim(0.5, 1.05)

if occlusions:
    fig.savefig(folder+'_fig6_bar_occ.pdf')
else:
    fig.savefig(folder+'_fig6_bar.pdf')


