# Edge Bench

## Table of Contents
* [Supported Models](#supported-models)
* [Pre-requisites](#pre-requisites)
* [How to Run](#how-to-run)

## Supported Models

## Pre-requisites
* Python >= 3.5
* Python Packages (Versions that we use.)
```bash
numpy===1.16.4

# PyTorch
torch===1.1.0
torchvision===0.2.2

# TensorFlow
tensorflow===1.13.1
Keras===2.2.4
```
#### PyTorch on Raspberry Pi
We follow [this](https://medium.com/hardware-interfacing/how-to-install-pytorch-v4-0-on-raspberry-pi-3b-odroids-and-other-arm-based-devices-91d62f2933c7)
tutorial to compile the PyTorch library from source on Raspberry Pi.

#### PyTorch on Nvidia Dev Boards
We use the default [JetPack](https://developer.nvidia.com/embedded/jetpack) 
library to setup both our dev boards (Nvidia TX2 and Nvidia Nano boards). Nvidia has its
pre-built PyTorch wheel [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/).
It has detailed instructions about how to install PyTorch on Nvidia Dev Boards.

#### TensorFlow on Raspberry Pi
We use pre-built wheel from [here](https://github.com/lhelontra/tensorflow-on-arm) for TensorFlow library on 
Raspberry Pi. 

#### TensorFlow on Nvidia Dev Boards
Same as PyTorch, Nvidia provides detailed instructions [here](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#install)
about how to install TensorFlow.

#### DarkNet
We compile the Darknet framework from source on the devices that we have. During compilation, you can 
specify the GPU support by setting the flag `GPU=1`. You can refer more complication details to the 
[website](https://pjreddie.com/darknet/install/).

## How to Run
#### PyTorch
```bash
cd pytorch
python execute.py --model [model name] --iteration [number of iterations] --cpu [Use CPU if set]
```

#### TensorFlow
```bash
cd tensorflow

# GPU
NVIDIA_VISIBLE_DEVICES=0 python execute.py --model [model name] --iteration [number of iterations]

# CPU
NVIDIA_VISIBLE_DEVICES= python execute.py --model [model name] --iteration [number of iterations]
```

#### DarkNet
We use the pre-existing model configurations in DarkNet code base to execute models.
```bash
./darknet classifier predict [base label data] [model config] [model weights] [inference data]
```
You can lookup more details [here](https://pjreddie.com/darknet/imagenet/).
