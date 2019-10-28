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

## How to Run
* PyTorch
```bash
cd pytorch
python execute.py --model [model name] --iteration [number of iterations] --cpu [Use CPU if set]
```

* TensorFlow
```bash
cd tensorflow

# GPU
NVIDIA_VISIBLE_DEVICES=0 python execute.py --model [model name] --iteration [number of iterations]

# CPU
NVIDIA_VISIBLE_DEVICES= python execute.py --model [model name] --iteration [number of iterations]
```
