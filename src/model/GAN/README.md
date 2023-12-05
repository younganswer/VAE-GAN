<h1 align='center'>Generative Adversarial Network</h1>
<br/><br/>

## Index

-   [Introduction](#introduction)
-   [Used hyperparameters](#used-hyperparameters)
-   [Dataset](#dataset)
-   [Model architecture](#model-architecture)
-   [Results](#results)

<br/><br/>

## Introduction

This is a PyTorch implementation of Generative Adversarial Network (GAN) based on the paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).<br/>
It is Deep Convolutional GAN (DCGAN) with Least Squares error (LSGAN) loss function based on the paper [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076).<br/>
It has unbanlanced layer and is pre-trained with VAE based on the paper [Unbalanced GANs](https://arxiv.org/abs/2002.02112).<br/>
The model is trained on the CelebA dataset.

<br/><br/>

## Used hyperparameters

-   Batch size: 128
-   Learning rate: 0.005
-   Optimizer: Adam
-   Number of epochs: 5
-   KL divergence weight in pre-training: 0.00025
-   Latent space dimension: 128
-   Loss function: Mean Squared Error (MSE)
-   Activation function: Leaky ReLU
-   Image size: 64 square
-   Label flipping step: 16
-   Label noise with Gaussian distribution: 0.1

<br/><br/>

## Dataset

The model is trained on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).<br/>
The dataset contains 202,599 face images of various celebrities.<br/>
The images are cropped and resized to 64 square images.

<br/><br/>

## Model architecture

-   Unit test with torchsummary

-   Generator

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 2048]         264,192
   ConvTranspose2d-2            [-1, 256, 4, 4]       1,179,904
       BatchNorm2d-3            [-1, 256, 4, 4]             512
         LeakyReLU-4            [-1, 256, 4, 4]               0
   ConvTranspose2d-5            [-1, 128, 8, 8]         295,040
       BatchNorm2d-6            [-1, 128, 8, 8]             256
         LeakyReLU-7            [-1, 128, 8, 8]               0
   ConvTranspose2d-8           [-1, 64, 16, 16]          73,792
       BatchNorm2d-9           [-1, 64, 16, 16]             128
        LeakyReLU-10           [-1, 64, 16, 16]               0
  ConvTranspose2d-11           [-1, 32, 32, 32]          18,464
      BatchNorm2d-12           [-1, 32, 32, 32]              64
        LeakyReLU-13           [-1, 32, 32, 32]               0
  ConvTranspose2d-14           [-1, 32, 64, 64]           9,248
      BatchNorm2d-15           [-1, 32, 64, 64]              64
        LeakyReLU-16           [-1, 32, 64, 64]               0
           Conv2d-17            [-1, 3, 64, 64]             867
             Tanh-18            [-1, 3, 64, 64]               0
================================================================
Total params: 1,842,531
Trainable params: 1,842,531
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 4.61
Params size (MB): 7.03
Estimated Total Size (MB): 11.64
----------------------------------------------------------------
```

<br/>

-   Discriminator

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             896
       BatchNorm2d-2           [-1, 32, 32, 32]              64
         LeakyReLU-3           [-1, 32, 32, 32]               0
            Conv2d-4           [-1, 64, 16, 16]          18,496
       BatchNorm2d-5           [-1, 64, 16, 16]             128
         LeakyReLU-6           [-1, 64, 16, 16]               0
            Conv2d-7            [-1, 128, 8, 8]          73,856
       BatchNorm2d-8            [-1, 128, 8, 8]             256
         LeakyReLU-9            [-1, 128, 8, 8]               0
           Conv2d-10            [-1, 256, 4, 4]         295,168
      BatchNorm2d-11            [-1, 256, 4, 4]             512
        LeakyReLU-12            [-1, 256, 4, 4]               0
           Linear-13                  [-1, 128]         524,416
        LeakyReLU-14                  [-1, 128]               0
           Linear-15                    [-1, 1]             129
================================================================
Total params: 913,921
Trainable params: 913,921
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 1.41
Params size (MB): 3.49
Estimated Total Size (MB): 4.94
----------------------------------------------------------------
```

<br/><br/>

## Results

You can see the results in the [notebook](../../../notebook/VAE/CelebA_64_square.ipynb).
