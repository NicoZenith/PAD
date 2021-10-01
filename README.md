# Generative adversarial networks in the brain


This repository proposes two models, inspired from the original GANs, to explain how the visual cortex can build semantic representations during wake and sleep. 

The first one is an extension of the DCGAN model (Radford et al., 2015), with a 2 feedforward networks, a discriminator and a generator. We extend this architecture by making the discriminator encode the images as a part of autoencoder with the generator (decoder), in addition to the usual binary classification (real/fake). G and D are trained by the classical GANs scheme, and we add reconstructions loss in the image space |x - G(D(x))| and latent space | z - D(G(z)) \| to make the D and G cycle consistent. We consider the former to happen during wakefulness, and the latter during sleep. 

The second one is an extension of the ALI model (Dumoulin et al., 2017) with a recurrent encoder-generator G and a feedforward discriminator D. G can encode an image by projecting it to the latent space or generate an image by projecting a random latent activity to the visible layer. To enable this process, the discriminator takes two input: one in the image space, the other one in the latent space. During Wakefulness, both image x and latent activity G(x,.) are learned to be classified as "real" by D, while G tries to make it classify as fake such that the latent activity becomes closer to the GAN-latent space. During dreaming,  both generated image G(.,z) and latent activity z are learned to be classified as "fake" by D, while G tries to make it classify as "real" such the activity at the visible layer looks realistic. 

In contrast with ALI, this model uses the same recurrent network G for both encoding and generation, where ALI uses two separate feedforward networks. The motivation with the proposed model is to match the structure of the visual cortex made of both feedforward and feedback connections, which can be exploited for both inference and dreaming. 
# PAD
