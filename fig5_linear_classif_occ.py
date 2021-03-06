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
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--niterC', type=int, default=20, help='number of epochs to train the classifier')
parser.add_argument('--nf', type=int, default=64, help='filters factor')
parser.add_argument('--drop', type=float, default=0, help='probably of dropping a patch')
parser.add_argument('--lrC', type=float, default=0.2, help='learning rate of the classifier, default=0.0002')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='baseline', help='folder to output images and model checkpoints')
parser.add_argument('--acc_file', default='accuracies_occ.pth', help='folder to output accuracies')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--tile_size', type=int, default=4, help='tile size for occlusions')
parser.add_argument('--gpu_id', type=str, default='0', help='The ID of the specified GPU')

opt, unknown = parser.parse_known_args()
print(opt)

# specify the gpu id if using only 1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

dir_files = './results/'+opt.dataset+'/'+opt.outf
dir_checkpoint = './checkpoints/'+opt.dataset+'/'+opt.outf
acc_file = opt.acc_file

try:
    os.makedirs(dir_files)
except OSError:
    pass
try:
    os.makedirs(dir_checkpoint)
except OSError:
    pass

drop_rate = opt.drop/100.0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if opt.dataset == 'cifar10':
    n_train = 50000
    n_test = 10000
elif opt.dataset == 'svhn':
    n_train = 73257
    n_test = 26032
    
# train dataset with occlusions (param drop_rate)
dataset, unorm, img_channels = get_dataset(dataset_name=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize, is_train=True, drop_rate=drop_rate, tile_size=opt.tile_size)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=n_train, shuffle=True, num_workers=int(opt.workers), drop_last=True)
# test dataset with occlusions (param drop_rate)
test_dataset, unorm, img_channels = get_dataset(dataset_name=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize, is_train=False, drop_rate=drop_rate, tile_size=opt.tile_size)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=n_test, shuffle=False, num_workers=int(opt.workers), drop_last=True)

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
num_classes = int(opt.num_classes)
batch_size = opt.batchSize

netG = Generator(ngpu, nz=nz, ngf=opt.nf, img_channels=img_channels)
netG.apply(weights_init)
netD = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels,  p_drop=opt.drop)
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
    d_losses = checkpoint.get('d_losses', [float('inf')])
    g_losses = checkpoint.get('g_losses', [float('inf')])
    r_losses = checkpoint.get('r_losses', [float('inf')])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')



train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

if os.path.exists(dir_files+'/' + acc_file):
    # Load data from last checkpoint
    print('Loading accuracies...')
    checkpoint = torch.load(dir_files+'/'+acc_file, map_location='cpu')
    train_accuracies = checkpoint.get('train_accuracies', [float('inf')])
    test_accuracies = checkpoint.get('test_accuracies', [float('inf')])
    train_losses = checkpoint.get('train_losses', [float('inf')])
    test_losses = checkpoint.get('test_losses', [float('inf')])
else:
    print('No accuracies found...')


n_epochs_c = opt.niterC
class_criterion = nn.CrossEntropyLoss()

print("Storing training representations ...")
image, label = next(iter(train_dataloader))
image, label = image.to(device), label.to(device)
netD.eval()
with torch.no_grad():
    latent_output, _ = netD(image)
    train_features = latent_output.cpu()
    train_labels = label.cpu().long()

print("Storing validation representations ...")
image, label = next(iter(test_dataloader))
image, label = image.to(device), label.to(device)
netD.eval()
with torch.no_grad():
    latent_output, _ = netD(image)
    test_features = latent_output.cpu()
    test_labels = label.cpu().long()

# create dataset of latent activities
linear_train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
linear_test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

linear_train_loader = torch.utils.data.DataLoader(linear_train_dataset, batch_size=batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
linear_test_loader = torch.utils.data.DataLoader(linear_test_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.workers, drop_last=True)

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
    print('No trained classifier detecte...')

# for epoch in range(n_epochs_c):

store_train_acc = []
store_test_acc = []
store_train_loss = []
store_test_loss = []

print("training on train set...")

for feature, label in linear_train_loader:
    feature, label = feature.to(device), label.to(device)
    classifier.eval()
    class_output = classifier(feature)
    class_err = class_criterion(class_output, label)
    # store train metrics
    train_acc = compute_acc(class_output, label)
    store_train_acc.append(train_acc)
    store_train_loss.append(class_err.item())


print("testing on test set...")
# compute test accuracy
for feature, label in linear_test_loader:
    feature, label = feature.to(device), label.to(device)
    classifier.eval()
    class_output = classifier(feature)
    class_err = class_criterion(class_output, label)
    # store test metrics
    test_acc = compute_acc(class_output, label)
    store_test_acc.append(test_acc)
    store_test_loss.append(class_err.item())

print('[%d/%d]  train_loss: %.4f  test_loss: %.4f  train_acc: %.4f  test_acc: %.4f'
          % (epoch, n_epochs_c, np.mean(store_train_loss), np.mean(store_test_loss), np.mean(store_train_acc), np.mean(store_test_acc)))



# train average metrics
train_accuracies.append(np.mean(store_train_acc))
train_losses.append(np.mean(store_train_loss))
# test average metrics
test_accuracies.append(np.mean(store_test_acc))
test_losses.append(np.mean(store_test_loss))

# do checkpointing
torch.save({
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies,
    'train_losses': train_losses,
    'test_losses': test_losses,
}, dir_files+'/' + acc_file)
print(f'Accuracies successfully saved.')


e = np.arange(0, len(train_accuracies))
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax1.plot(e, train_losses, label='train loss')
ax1.plot(e, test_losses, label='test loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.set_title('losses with uns. training')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.plot(e, train_accuracies, label='train acc')
ax2.plot(e, test_accuracies, label='test acc')
ax2.set_ylim(0, 100)
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy (%)')
ax2.set_title('accuracy with uns. training')
ax2.legend()

fig.savefig(dir_files + '/linear_classif_occ.pdf')




