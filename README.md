# pytorch-gd-uap
Code for the paper [Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations](https://arxiv.org/abs/1801.08092) by Mopuri et al., 2018.

This repository **depends on PyTorch**, but you can refer to [the original repository](https://github.com/val-iisc/GD-UAP) if you prefer TensorFlow.

## Overview
A universal adversarial perturbation (UAP) is an image-agnostic perturbation vector that, when added to any image, leads a classifier to change its classification of the image.

The main algorithm for optimizing a UAP is in `gduap.py`. The range prior and the data prior, described in the original paper, are not implemented here.


## Usage Instructions

### 1. Preparation
1. Install dependencies listed in `requirements.txt`. Note that not all the dependencies are required, but the main modules are **torch 1.6.0** and **torchvision 0.7.0**.
2. In `gduap.py` set the variables `TORCH_HUB_DIR`, `IMAGENET_VAL_DIR`, and `VOC_VAL_DIR` at the top of the file.
- `TORCH_HUB_DIR` should be the directory where you'd like PyTorch pretrained model parameters to be saved. More info: [torch.hub documentation](https://pytorch.org/docs/stable/hub.html#torch.hub.get_dir)
- `IMAGENET_VAL_DIR` should be the directory containing `ILSVRC2012_devkit_t12.tar.gz`. More info: [ImageNet](http://image-net.org/index)
- `VOC_VAL_DIR` should be the directory containing `VOCdevkit`. More info: [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)


### 2. Training
Note: you can skip to Section 3 if you don't intend on training new UAPs. Example perturbation vectors are provided in the `perturbations/` folder for evaluation.

To optimize a UAP for a VGG-16:
```
python3 train.py --model vgg16 --id 12345
```
By default, this will use Pascal VOC 2012 Validation Set as the "substitute dataset" described in the paper. The final evaluation is performed on the ILSVRC 2012 Validation Set of 50k images. The `id` option can be used to uniquely prefix the UAP files that are saved to the `perturbations/` folder. Of course, you can call `python3 train.py --help` to see all the options.

### 3. Testing
After a UAP is optimized using the `train.py` script, evaluation is automatically performed on the ILSVRC 2012 Validation Set. You can also refer to `Sample (Evaluation).ipynb` to understand how fooling rate is evaluated.

## Included Notebooks

- `Plot.ipynb` plots perturbation vectors that have been previously optimized on VGG-16, VGG-19, GoogLeNet, ResNet-50, and ResNet-152.
- `Sample (Evaluation).ipynb` is a sample of the steps that should be taken to evaluate a UAP's fooling rate. Namely, we must load a model to do the classification, load a dataset on which to perform evaluation, and load a UAP to perturb the images.
