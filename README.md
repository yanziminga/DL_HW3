# DL_HW3
# DCGAN and WGAN Implementation
## Introduction
Train a discriminator/generator pair on CIFAR10 dataset utilizing techniques from DCGAN,  Wasserstein GANs

## Requirement
* absl-py 
* scipy 
* tensorflow 
* tensorboardX 
* torch==1.8.1 
* torchvision==0.9.1 tqdm

## Test the model performance
### Training
* Download the test dataset(stats folder) from the link provided in the report
* Change "Generate" to False in DCGAN.txt and WGAN.txt
* DCGAN
```c
python dcgan.py --flagfile ./config/DCGAN.txt
```
* WGAN
```c
python wgan.py --flagfile ./config/WGAN.txt
```

### Testing
* Download the trained model(logs folder) from the link provided in the report
* Change "Generate" to True in DCGAN.txt and WGAN.txt
* DCGAN
```c
python dcgan.py --flagfile ./config/DCGAN.txt
```
* WGAN
```c
python wgan.py --flagfile ./config/WGAN.txt
```
## need to be downloaded
* stats folder (includes test data)
* logs (include trained models)
