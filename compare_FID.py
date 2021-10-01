from __future__ import print_function
import argparse
import os
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from utils import *
from scipy import stats, optimize, interpolate


folder = './results/'
dataset = 'svhn'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
idx = ['', '1', '2', '3']
dir_files = []
for i in idx:
    dir_files.append('./results/'+dataset+'/fid_eval'+i)


all_fid_NREM = np.zeros((len(dir_files), 2))
all_fid_REM = np.zeros((len(dir_files), 2))

for i in range(len(dir_files)):
    print(i)
    print('Loading accuracies...')
    all_fid_NREM[i] = torch.load(dir_files[i]+'/'+ 'frechet_dist.pth', map_location=device).get('frechet_dist_NREM', [float('inf')])
    all_fid_REM[i] = torch.load(dir_files[i]+'/'+ 'frechet_dist.pth', map_location=device).get('frechet_dist_REM', [float('inf')])

def mean_and_err(array, axis=0):
    mean = array.mean(axis=0)
    sem = stats.sem(array, axis=axis)
    return mean, sem



x = ['Early training', 'Late training']
fid_NREM = [ mean_and_err(all_fid_NREM[:,k])[0] for k in range(2) ]
fid_REM = [ mean_and_err(all_fid_REM[:,k])[0] for k in range(2) ]
err_NREM = [ mean_and_err(all_fid_NREM[:,k])[1] for k in range(2) ]
err_REM = [ mean_and_err(all_fid_REM[:,k])[1] for k in range(2) ]
colors_NREM = ['dimgray', 'dimgray']
colors_REM =  ['lightgray', 'lightgray']
edgecolors = [ 'black', 'black']

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
ind = np.arange(2)  # the x locations for the groups
width = 0.35
ax.bar(ind - width/2, fid_NREM, yerr=err_NREM, linewidth=1, edgecolor=edgecolors, color=colors_NREM, label='NREM', width=width, capsize=5)
ax.bar(ind + width/2, fid_REM, yerr=err_REM, linewidth=1, edgecolor=edgecolors, color=colors_REM, label='REM', width=width, capsize=5)

ax.set_ylabel("Fréchet inception distance", fontsize=16, labelpad=10)
ax.set_xticklabels(('Early training', 'Late training'))
#if dataset == 'cifar10':
#    ax.set_title('CIFAR-10', fontsize=16, fontweight='bold', y=0.95)
#else:
#    ax.set_title('SVHN', fontsize=16, fontweight='bold', y=0.95)
ax.set_xticks(ind)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for axis in 'left', 'bottom':
  ax.spines[axis].set_linewidth(1.5)
ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
ax.set_ylim(0, 300)
plt.tight_layout()
# set the parameters for both axis: label size in font points, the line tick line
# width and length in pixels
ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
#plt.tight_layout()
#fig.subplots_adjust(top=.95)

ax.legend(loc="best", frameon=False, fontsize=16)

fig.savefig(folder+dataset+'_fig3_fid.pdf', bbox_inches="tight")

