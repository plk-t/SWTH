# Swin Transformer-based Supervised Hashing (SWTH)

The official implementation of **Swin Transformer-based Supervised Hashing**

**Requirements**

* Linux with Python >= 3.7
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.7.0
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA 11.0


## Getting Started

The CIFAR10 and ImageNet datasets files could be obtained from \
https://www.cs.toronto.edu/~kriz/cifar.html \
https://image-net.org/index.php

Train on CIFAR10

Trained model will be saved in 'output/CIFAR10/'

```
python main.py --cfg ./configs/swin_config_cifar.yaml --batch_size 32
```



Train on ImageNet

Trained model will be saved in 'output/imagenet/'

```
python main.py --cfg ./configs/swin_config_imgnet.yaml --batch_size 32
```
