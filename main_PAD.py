
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
parser.add_argument('--outf', default='dd', help='folder to output images and model checkpoints')
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True)

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
batch_size = opt.batchSize

# setup networks
netG = Generator(ngpu, nz=nz, ngf=opt.nf, img_channels=img_channels)
netG.apply(weights_init)
netD = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels,  p_drop=opt.drop)
netD.apply(weights_init)
# send to GPU
netD.to(device)
netG.to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
d_losses = []
g_losses = []
r_losses_real = []
r_losses_fake = []
kl_losses = []


if os.path.exists(dir_checkpoint+'/trained.pth') and opt.is_continue:
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location=torch.device('cpu'))
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    optimizerG.load_state_dict(checkpoint['g_optim'])
    optimizerD.load_state_dict(checkpoint['d_optim'])
    d_losses = checkpoint.get('d_losses', [float('inf')])
    g_losses = checkpoint.get('g_losses', [float('inf')])
    r_losses_real = checkpoint.get('r_losses_real', [float('inf')])
    r_losses_fake = checkpoint.get('r_losses_fake', [float('inf')])
    kl_losses = checkpoint.get('kl_losses', [float('inf')])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')


# loss functions
dis_criterion = nn.BCELoss() # discriminator
rec_criterion = nn.MSELoss() # reconstruction

# tensor placeholders
dis_label = torch.zeros(opt.batchSize, dtype=torch.float32, device=device)
real_label_value = 1.0
fake_label_value = 0

eval_noise = torch.randn(batch_size, nz, device=device)


#torch.autograd.set_detect_anomaly(True)

for epoch in range(len(d_losses), opt.niter):

    store_loss_D = []
    store_loss_G = []
    store_loss_R_real = []
    store_loss_R_fake = []
    store_norm = []
    store_kl = []

    for i, data in enumerate(dataloader, 0):

        ############################
        # Wake phase (A1)
        ###########################
        # Discrimination wake
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        real_image, label = data
        real_image, label = real_image.to(device), label.to(device)
        latent_output, dis_output = netD(real_image)
        latent_output_noise = latent_output + opt.epsilon*torch.randn(batch_size, nz, device=device) # noise transformation
        dis_label[:] = real_label_value  # should be classified as real
        dis_errD_real = dis_criterion(dis_output, dis_label)
        #dis_errD_real = 0.5 * torch.mean((dis_output - dis_label)**2)
        if opt.R > 0.0:  # if GAN learning occurs
            (dis_errD_real).backward(retain_graph=True)

        # KL divergence regularization
        kl = kl_loss(latent_output)
        (kl).backward(retain_graph=True)
        
        # reconstruction Real data space
        reconstructed_image = netG(latent_output_noise, reverse=False)
        rec_real = rec_criterion(reconstructed_image, real_image)
        if opt.W > 0.0:
            (opt.W*rec_real).backward()
        optimizerD.step()
        optimizerG.step()
        # compute the mean of the discriminator output (between 0 and 1)
        D_x = dis_output.cpu().mean()
        latent_norm = torch.mean(torch.norm(latent_output.squeeze(), dim=1)).item()

        ###########################
        # NREM reconstruction of Wake memory (B1)
        ##########################
        optimizerD.zero_grad()
        latent_z = latent_output.detach()
        if opt.N > 0.0:
            with torch.no_grad():
                nrem_image = netG(latent_z)
                occlusion = Occlude(drop_rate=random.random(), tile_size=random.randint(1,8))
                occluded_nrem_image = occlusion(nrem_image, d=1)
            latent_recons_dream, _ = netD(occluded_nrem_image)
            rec_fake = rec_criterion(latent_recons_dream, latent_output.detach())
            (opt.N * rec_fake).backward()
            optimizerD.step()


        ###########################
        #Adversarial Phase (B2)
        ##########################

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lmbd = opt.lmbd
        if i==0:
            latent_z = latent_output.detach()
        else:
            latent_z = lmbd*latent_output.detach() + (1-lmbd)*old_latent_output
        dreamed_image_adv = netG(latent_z, reverse=True) # activate plasticity switch
        latent_recons_dream, dis_output = netD(dreamed_image_adv)
        rec_fake = rec_criterion(latent_recons_dream, latent_z)
        dis_label[:] = fake_label_value # should be classified as fake
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        #dis_errD_fake = 0.5 * torch.mean((dis_output - dis_label) ** 2)
        if opt.R > 0.0: # if GAN learning occurs
            dis_errD_fake.backward(retain_graph=True)
            optimizerD.step()
            optimizerG.step()
        dis_errG = - dis_errD_fake

        D_G_z1 = dis_output.cpu().mean()

        old_latent_output = latent_output.detach()
        
        ###########################
        # Compute average losses
        ###########################
        store_loss_G.append(dis_errG.item())
        store_loss_D.append((dis_errD_fake + dis_errD_real).item())
        store_loss_R_real.append(rec_real.item())
        store_loss_R_fake.append(rec_fake.item())
        store_norm.append(latent_norm)
        store_kl.append(kl.item())
        


        if i % 200 == 0 and i>1:
            print('[%d/%d][%d/%d]  Loss_D: %.4f  Loss_G: %.4f  Loss_R_real: %.4f  Loss_R_fake: %.4f  D(x): %.4f  D(G(z)): %.4f  latent_norm : %.4f  '
                % (epoch, opt.niter, i, len(dataloader),
                    np.mean(store_loss_D), np.mean(store_loss_G), np.mean(store_loss_R_real), np.mean(store_loss_R_fake), D_x, D_G_z1, np.mean(latent_norm) ))
            compare_img_rec = torch.zeros(batch_size * 2, real_image.size(1), real_image.size(2), real_image.size(3))
            with torch.no_grad():
                reconstructed_image = netG(latent_output)
            compare_img_rec[::2] = real_image
            compare_img_rec[1::2] = reconstructed_image
            vutils.save_image(unorm(compare_img_rec[:128]), '%s/recon_%03d.png' % (dir_files, epoch), nrow=8)
            fake = unorm(dreamed_image_adv)
            vutils.save_image(fake[:64].data, '%s/fake_%03d.png' % (dir_files, epoch), nrow=8)
            

    d_losses.append(np.mean(store_loss_D))
    g_losses.append(np.mean(store_loss_G))
    r_losses_real.append(np.mean(store_loss_R_real))
    r_losses_fake.append(np.mean(store_loss_R_fake))
    kl_losses.append(np.mean(store_kl))
    save_fig_losses(epoch, d_losses, g_losses, r_losses_real, r_losses_fake, kl_losses, None, None,  dir_files)

    # do checkpointing
    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        'g_optim': optimizerG.state_dict(),
        'd_optim': optimizerD.state_dict(),
        'd_losses': d_losses,
        'g_losses': g_losses,
        'r_losses_real': r_losses_real,
        'r_losses_fake': r_losses_fake,
        'kl_losses': kl_losses,
    }, dir_checkpoint+'/trained.pth')
    
    if epoch ==1:
            torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        }, dir_checkpoint+'/trained2.pth')

    print(f'Model successfully saved.')
