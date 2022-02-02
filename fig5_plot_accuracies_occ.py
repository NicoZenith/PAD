from __future__ import print_function
import argparse
import os
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from scipy import stats, optimize, interpolate

dataset = 'cifar10'
folder = './results/'

acc_file = 'accuracies_levels.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
idx = ['', '1', '2', '3']
dir_files_1, dir_files_2, dir_files_3, dir_files_4 = [],[],[],[]
#for i in idx:
#    dir_files_1.append('./results/'+dataset+'/model_mix_wr'+i)
#    dir_files_2.append('./results/' + dataset + '/model_mix_noreplay' + i)

for i in idx:
    dir_files_1.append('./results/'+dataset+'/model_mix_wnr'+i)
    dir_files_2.append('./results/' + dataset + '/model_mix_noreplay' + i)


accuracies_1 = np.zeros((len(dir_files_1), 11))
accuracies_2 = np.zeros((len(dir_files_2), 11))
accuracies_3 = np.zeros((len(dir_files_3), 11))
accuracies_4 = np.zeros((len(dir_files_4), 11))

for i in range(len(dir_files_1)):
    print(i)
    print('Loading accuracies...')
    accuracies_1[i] = torch.load(dir_files_1[i]+'/'+acc_file, map_location=device).get('test_accuracies', [float('inf')])
    accuracies_2[i] = torch.load(dir_files_2[i] + '/' + acc_file, map_location=device).get('test_accuracies', [float('inf')])

def mean_and_err(array, axis=0):
    mean = array.mean(axis=0)
    sem = stats.sem(array, axis=axis)
    return mean, sem


probas = np.arange(0, 110, 10) # stores all probabilities for drop rates
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)


#
#ax.plot(probas, mean_and_err(accuracies_1)[0], color='black', marker='o', label='PAD')
#ax.plot(probas, mean_and_err(accuracies_2)[0], color='darkorange', marker='o', label='w/o NREM')

ax.plot(probas, mean_and_err(accuracies_1)[0], color='black', marker='o', label='NREM with replay (PAD)')
ax.plot(probas, mean_and_err(accuracies_2)[0], color='darkred', marker='o', label='NREM with mix')


ax.set_xlabel('Occlusion intensity (%)', fontsize=14, labelpad=5)
if dataset=='cifar10':
    ax.set_ylabel('Linear separability (%)', fontsize=14, labelpad=10)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
# set the axis line width in pixels
for axis in 'left', 'bottom':
  ax.spines[axis].set_linewidth(1.5)
# set the parameters for both axis: label size in font points, the line tick line
# width and length in pixels
ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
plt.tight_layout()

ax.set_ylim(0, 85)

if dataset=='cifar10':
    ax.legend(loc="best", frameon=False, fontsize=14)

#fig.savefig(folder+dataset+'_fig5_full_levels.pdf')
fig.savefig(folder+dataset+'_supp_levels.pdf')
