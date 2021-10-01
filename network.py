import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.models as models
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf=64, img_channels=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.bias = True

        # input is Z, going into a convolution
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=1, padding=0, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ngf*4) x 4 x 4
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ngf*2) x 8 x 8
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # state size. (ngf) x 16 x 16
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, img_channels, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.Tanh()
        )

    def forward(self, input, reverse=True):
        fc1 = input.view(input.size(0), input.size(1), 1, 1)
        tconv1 = self.tconv1(fc1)
        tconv2 = self.tconv2(tconv1)
        tconv3 = self.tconv3(tconv2)
        output = self.tconv4(tconv3)
        if reverse:
            output = grad_reverse(output)
        return output



class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse.apply(x)
    
    

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)





class Discriminator(nn.Module):
    def __init__(self, ngpu, nz, ndf=64, img_channels=3, p_drop=0.0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.bias = True


        # input is (3) x 32 x 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels, ndf, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # state size. (64) x 16 x 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf*2) x 8 x 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )


        # # state size. (ndf*4) x 4 x 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, nz, kernel_size=4,stride=2, padding=0, bias=self.bias),
            Flatten()
        )
        
        self.dis = nn.Sequential(
             nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=0, bias=self.bias),
             Flatten()
        )

        # softmax and sigmoid 
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        fc_dis = self.sigmoid(self.dis(conv3))
        fc_enc = self.conv4(conv3)
        realfake = fc_dis.view(-1, 1).squeeze(1)
        return fc_enc, realfake
        






class OutputClassifier(nn.Module):

    def __init__(self, nz, ni=32, num_classes=10):
        super(OutputClassifier, self).__init__()

        self.fc_classifier = nn.Sequential(
            nn.Linear(nz, num_classes, bias=True),
        )
        self.softmax = nn.Softmax()

    def forward(self, input):
        classes = self.fc_classifier(input)
        return classes




class InputClassifier(nn.Module):

    def __init__(self, input_dim, num_classes=10):
        super(InputClassifier, self).__init__()
        self.fc_classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes, bias=True),
        )

    def forward(self, input):
        out = input.view(input.size(0), -1)  # convert batch_size x 28 x 28 to batch_size x (28*28)
        out = self.fc_classifier(out)  # Applies out = input * A + b. A, b are parameters of nn.Linear that we want to learn
        return out






class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
