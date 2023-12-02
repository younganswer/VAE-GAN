<h1 align='center'>Variational AutoEncoder</h1>
</br></br>

## Index

-   [Introduction](#introduction)
-   [Used hyperparameters](#used-hyperparameters)
-   [Dataset](#dataset)
-   [Model architecture](#model-architecture)

</br></br>

## Introduction

<div>
	This is a PyTorch implementation of Variational AutoEncoder (VAE) based on the paper [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
</div>
<div>
	The model is trained on the CelebA dataset.
</div>

</br></br>

## Used hyperparameters

-   Batch size: 128
-   Learning rate: 0.005
-   Optimizer: Adam
-   Number of epochs: 5
-   Loss function: Mean Squared Error (MSE)

</br></br>

## Dataset

The model is trained on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The dataset contains 202,599 face images of various celebrities. The images are cropped and resized to 64 square images.

</br></br>

## Model architecture

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
           Conv2d-13            [-1, 512, 2, 2]       1,180,160
      BatchNorm2d-14            [-1, 512, 2, 2]           1,024
        LeakyReLU-15            [-1, 512, 2, 2]               0
           Linear-16                   [-1, 10]          20,490
           Linear-17                   [-1, 10]          20,490
           Linear-18                 [-1, 2048]          22,528
  ConvTranspose2d-19            [-1, 256, 4, 4]       1,179,904
      BatchNorm2d-20            [-1, 256, 4, 4]             512
        LeakyReLU-21            [-1, 256, 4, 4]               0
  ConvTranspose2d-22            [-1, 128, 8, 8]         295,040
      BatchNorm2d-23            [-1, 128, 8, 8]             256
        LeakyReLU-24            [-1, 128, 8, 8]               0
  ConvTranspose2d-25           [-1, 64, 16, 16]          73,792
      BatchNorm2d-26           [-1, 64, 16, 16]             128
        LeakyReLU-27           [-1, 64, 16, 16]               0
  ConvTranspose2d-28           [-1, 32, 32, 32]          18,464
      BatchNorm2d-29           [-1, 32, 32, 32]              64
        LeakyReLU-30           [-1, 32, 32, 32]               0
  ConvTranspose2d-31            [-1, 3, 64, 64]             867
      BatchNorm2d-32            [-1, 3, 64, 64]               6
             Tanh-33            [-1, 3, 64, 64]               0
================================================================
Total params: 3,203,101
Trainable params: 3,203,101
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 3.16
Params size (MB): 12.22
Estimated Total Size (MB): 15.42
----------------------------------------------------------------
```