import torch
import numpy as np
# plotting
import matplotlib
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset
from scipy import linalg


matplotlib.use('Agg')
import matplotlib.pyplot as plt

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_dataset(dataset_name, dataroot, imageSize, is_train=True, drop_rate=0.0, tile_size=32):
    if dataset_name == 'cifar10':
        dataset = dset.CIFAR10(
            train=is_train,
            root=dataroot, download=False,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                Occlude(drop_rate=drop_rate, tile_size=tile_size),
            ]))
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img_channels = 3

    elif dataset_name == 'svhn':
        if is_train:
            split = 'train'
        else:
            split = 'test'
        dataset = dset.SVHN(
            root=dataroot, download=False,
            split = split,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                Occlude(drop_rate=drop_rate, tile_size=tile_size)
            ]))
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img_channels = 3

    elif dataset_name == 'mnist':
        dataset = dset.MNIST(
            train=is_train,
            root=dataroot, download=False,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                Occlude(drop_rate=drop_rate, tile_size=tile_size)
            ])
        )
        unorm = UnNormalize(mean=(0.5,), std=(0.5,))
        img_channels = 1
    elif dataset_name == 'fashion':
        dataset = dset.FashionMNIST(
            train=is_train,
            root=dataroot, download=False,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                Occlude(drop_rate=drop_rate, tile_size=tile_size)
            ]))
        unorm = UnNormalize(mean=(0.5,), std=(0.5,))
        img_channels = 1
    else:
        raise NotImplementedError("No such dataset {}".format(dataset_name))

    assert dataset
    return dataset, unorm, img_channels



# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


def get_latent(dim_latent, batch_size, device):
    latent_z = np.random.normal(0, 1, (batch_size, dim_latent))  # generate random labels
    latent_z = torch.tensor(latent_z, dtype=torch.float32, device=device)
    latent_z = latent_z.view(batch_size, dim_latent, 1, 1)
    return latent_z


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensorBatch):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for i in range(len(tensorBatch)):
            for j in range(len(tensorBatch[i])):
                tensorBatch[i][j].mul_(self.std[j]).add_(self.mean[j])
            # The normalize code -> t.sub_(m).div_(s)
        return tensorBatch


def save_fig_losses(epoch, d_losses, g_losses, r_losses_real, r_losses_fake, kl_losses, fid_NREM, fid_REM,  dir_files):
    e = np.arange(0, epoch+1)
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    if g_losses is not None:
        ax1.plot(e, g_losses, label='generator (REM)')
    if d_losses is not None:
        ax1.plot(e, d_losses, color='green', label='discriminator (Wake, REM)')
    #ax1.set_ylim(0, 10)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title('losses with training')
    if r_losses_real is not None:
        ax1.plot(e, r_losses_real, color='orange', label='data rec. (Wake)')
    if r_losses_fake is not None:
        ax1.plot(e, r_losses_fake, color='magenta', label='latent rec. (NREM)')
    if kl_losses is not None:
        ax1.plot(e, kl_losses, color='brown', label='KL div. (Wake)')
    ax1.legend()
    
    if fid_NREM is not None and fid_REM is not None:
        ax2 = fig.add_subplot(122)
        ax2.plot(e, fid_NREM, color='darkorange', label='FID NREM')
        ax2.plot(e, fid_REM, color='magenta', label='FID REM')
        ax2.legend()
    fig.savefig(dir_files+'/losses.pdf')


def save_fig_trainval(epoch, all_losses, all_accuracies, dir_files):
    e = np.arange(0, epoch+1)
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax1.plot(e, all_losses['train'], label='train loss')
    ax1.plot(e, all_losses['val'], label='validation loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(e, all_accuracies['train'], label='train accuracy')
    ax2.plot(e, all_accuracies['val'], label='val accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.legend()
    fig.savefig(dir_files + '/trainval.pdf')



class Occlude(object):
    def __init__(self, drop_rate=0.0, tile_size=7):
        self.drop_rate = drop_rate
        self.tile_size = tile_size

    def __call__(self, imgs, d=0):
        imgs_n = imgs.clone()
        if d==0:
            device='cpu'
        else:
            device = imgs.get_device()
            if device ==-1:
                device = 'cpu'
        mask = torch.ones((imgs_n.size(d), imgs_n.size(d+1), imgs_n.size(d+2)), device=device)  # only ones = no mask
        i = 0
        while i < imgs_n.size(d+1):
            j = 0
            while j < imgs_n.size(d+2):
                if np.random.rand() < self.drop_rate:
                    for k in range(mask.size(0)):
                        mask[k, i:i + self.tile_size, j:j + self.tile_size] = 0  # set to zero the whole tile
                j += self.tile_size
            i += self.tile_size
        
        imgs_n = imgs_n * mask  # apply the mask to each image
        return imgs_n



def kl_loss(latent_output):
    m = torch.mean(latent_output, dim=0)
    s = torch.std(latent_output, dim=0)
    
    kl_loss = torch.mean((s ** 2 + m ** 2) / 2 - torch.log(s) - 1/2)
    return kl_loss


def mean_and_sem(array, color=None, axis=0):
    mean = array.mean(axis=0)
    sem_plus = mean + stats.sem(array, axis=axis)
    sem_minus = mean - stats.sem(array, axis=axis)
    if color is not None:
        ax.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, color=color, alpha=0.5)
    else:
        ax.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.5)
    return mean




def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]
    

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    return act 
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    
    
    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
            
            
            
def calculate_frechet(inception_real,inception_fake,model, return_statistics=False):
     mu_1 = np.mean(inception_real, axis=0)
     mu_2 = np.mean(inception_fake, axis=0)
     std_1 = np.cov(inception_real, rowvar=False)
     std_2 = np.cov(inception_fake, rowvar=False)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)

     return fid_value
     
     
    
