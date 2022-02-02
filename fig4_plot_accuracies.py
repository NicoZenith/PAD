from __future__ import print_function
import argparse
import os
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from utils import *
from scipy import stats, optimize, interpolate



dataset = 'svhn'
occlusions = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
idx = ['', '1', '2', '3']
dir_files_1, dir_files_2, dir_files_3, dir_files_4, dir_files_5 = [],[],[],[],[]
for i in idx:
    dir_files_1.append('./results/'+dataset+'/model_mix_wnr'+i)
    dir_files_2.append('./results/' + dataset + '/order_wnr' + i)
    dir_files_3.append('./results/' + dataset + '/model_wnr' + i)
    dir_files_4.append('./results/' + dataset + '/model100_wnr' + i)
    dir_files_5.append('./results/' + dataset + '/nomix100_wnr' + i)
    
#for i in idx:
#    dir_files_1.append('./results/'+dataset+'/model_w'+i)
#    dir_files_2.append('./results/' + dataset + '/model_wn' + i)
#    dir_files_3.append('./results/' + dataset + '/model_wr' + i)
#    dir_files_4.append('./results/' + dataset + '/model_wnr' + i)

if not occlusions:
    acc_file = 'accuracies.pth'
else:
    acc_file = 'accuracies_occ.pth'

if dataset == 'cifar10':
    dim = 50
    baseline = 29
else:
    dim = 50
    baseline = 19
    

accuracies_1 = np.zeros((len(dir_files_1), dim+1))
accuracies_2 = np.zeros((len(dir_files_2), dim+1))
accuracies_3 = np.zeros((len(dir_files_3), dim+1))
accuracies_4 = np.zeros((len(dir_files_4), dim+1))
accuracies_5 = np.zeros((len(dir_files_4), dim+1))

for i in range(len(dir_files_1)):
    print(i)
    print('Loading accuracies...')
    accuracies_1[i] = [baseline] + torch.load(dir_files_1[i]+'/'+acc_file, map_location=device).get('test_accuracies', [float('inf')])[:dim]
    accuracies_2[i] = [baseline] + torch.load(dir_files_2[i] + '/' + acc_file, map_location=device).get('test_accuracies', [float('inf')])[:dim]
    accuracies_3[i] = [baseline] + torch.load(dir_files_3[i] + '/' + acc_file, map_location=device).get('test_accuracies', [float('inf')])[:dim]
    accuracies_4[i] = [baseline] + torch.load(dir_files_4[i] + '/' + acc_file, map_location=device).get('test_accuracies', [float('inf')])[:dim]
    accuracies_5[i] = [baseline] + torch.load(dir_files_5[i] + '/' + acc_file, map_location=device).get('test_accuracies', [float('inf')])[:dim]
    

def mean_and_sem(array, color=None, return_sem=False, axis=0):
    mean = array.mean(axis=0)
    sem_plus = mean + stats.sem(array, axis=axis)
    sem_minus = mean - stats.sem(array, axis=axis)
    if color is not None:
        ax.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, color=color, alpha=0.5)
    else:
        ax.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.5)
    #return mean, stats.sem(array, axis=axis)
    return mean
        
    



epochs = np.arange(0, dim+1, 1) # stores all probabilities for drop rates
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

#ax.plot(epochs, mean_and_sem(accuracies_1, axis=0, color='black'), color = 'black', label='$\lambda^\prime (\lambda z_1 + (1-\lambda)z_2) + (1-\lambda^\prime )\epsilon$')
#ax.plot(epochs, mean_and_sem(accuracies_2, axis=0, color='green'), color = 'green', label='$\epsilon$')
#ax.plot(epochs, mean_and_sem(accuracies_3, axis=0, color='red'), color = 'red', label='$\lambda z_1 + (1-\lambda)z_2$ ')

#ax.plot(epochs, mean_and_sem(accuracies_1, axis=0, color='black'), color = 'black', label='Wake-NREM-REM')
#ax.plot(epochs, mean_and_sem(accuracies_2, color='peru', axis=0), color ='peru', label='Wake-REM-NREM')


## plot all conditions
ax.plot(epochs, mean_and_sem(accuracies_1, axis=0, color='black'), color = 'black', label='PAD')
ax.plot(epochs, mean_and_sem(accuracies_2, axis=0, color='blueviolet'), color = 'blueviolet', label='w/o memory mix')
ax.plot(epochs, mean_and_sem(accuracies_3, axis=0, color='magenta'), color = 'magenta', label='w/o REM')
#ax.plot(epochs, mean_and_sem(accuracies_3, axis=0, color='darkorange'), color = 'darkorange', label='w/o NREM')
#ax.plot(epochs, mean_and_sem(accuracies_1, axis=0, color='silver'), color = 'silver', label='Wake only')

# numerical values for table in supplementary
#print(mean_and_sem(accuracies_4)[0][50], mean_and_sem(accuracies_4)[1][50])
#print(mean_and_sem(accuracies_5)[0][50], mean_and_sem(accuracies_5)[1][50])
#print(mean_and_sem(accuracies_2)[0][50], mean_and_sem(accuracies_2)[1][50])
#print(mean_and_sem(accuracies_3)[0][50], mean_and_sem(accuracies_3)[1][50])
#print(mean_and_sem(accuracies_1)[0][50], mean_and_sem(accuracies_1)[1][50])


ax.set_xlabel('Learning epochs', fontsize=14, labelpad=5)
if dataset=='cifar10':
    ax.set_ylabel('Linear separability (%)', fontsize=14, labelpad=10)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for axis in 'left', 'bottom':
  ax.spines[axis].set_linewidth(1.5)
# set the parameters for both axis: label size in font points, the line tick line
# width and length in pixels
ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
plt.tight_layout()
plt.tight_layout()

ax.set_ylim(0, 85)

if dataset=='cifar10' and not occlusions:
    ax.legend(loc="lower right", frameon=False, fontsize=14)

#fig.savefig('./results/'+dataset+'_fig4.pdf')
fig.savefig('./results/'+dataset+'_supp_linear_order.pdf')
