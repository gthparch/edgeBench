import torch
from torch import cuda

from torchvision.models import resnet, alexnet, vgg

import argparse
from timeit import default_timer as timer

from models import xception, inception, mobilenet, c3d, yolo3, cifarnet, deepspeech, resnet50_ssd

args = argparse.ArgumentParser()
args.add_argument('--model', type=str, help='Model name.')
args.add_argument('--cpu', default=False, type=bool, help='Using CPU or not.')
args.add_argument('-i', '--iteration', default=1000, type=int, help='Test iteration.')
parser = args.parse_args()


def execute(model_f, name, data_shape, gpu):
    device = 'cuda' if cuda.is_available() and gpu else 'cpu'
    net = model_f().to(device)

    start = timer()
    with torch.no_grad():
        for _ in range(parser.iteration):
            data = torch.randn(data_shape).to(device)
            net(data)
    print(f'{name} inference time (sec): {(timer() - start) / parser.iteration:5f}')


try_use_gpu = not parser.cpu
if parser.model == 'resnet-18':
    data_shape = [1, 3, 224, 224]
    execute(resnet.resnet18, parser.model, data_shape, try_use_gpu)
elif parser.model == 'resnet-50':
    data_shape = [1, 3, 224, 224]
    execute(resnet.resnet50, parser.model, data_shape, try_use_gpu)
elif parser.model == 'resnet-101':
    data_shape = [1, 3, 224, 224]
    execute(resnet.resnet101, parser.model, data_shape, try_use_gpu)
elif parser.model == 'alexnet':
    data_shape = [1, 3, 224, 224]
    execute(alexnet, parser.model, data_shape, try_use_gpu)
elif parser.model == 'vgg16':
    data_shape = [1, 3, 224, 224]
    execute(vgg.vgg16, parser.model, data_shape, try_use_gpu)
elif parser.model == 'vgg19':
    data_shape = [1, 3, 224, 224]
    execute(vgg.vgg19, parser.model, data_shape, try_use_gpu)
elif parser.model == 'vggs-32':
    data_shape = [1, 3, 32, 32]
    execute(vgg.vgg11, parser.model, data_shape, try_use_gpu)
elif parser.model == 'vggs-224':
    data_shape = [1, 3, 224, 224]
    execute(vgg.vgg11, parser.model, data_shape, try_use_gpu)
elif parser.model == 'xception':
    data_shape = [1, 3, 229, 229]
    execute(xception.Xception, parser.model, data_shape, try_use_gpu)
elif parser.model == 'inception':
    data_shape = [1, 3, 229, 229]
    execute(inception.InceptionV4, parser.model, data_shape, try_use_gpu)
elif parser.model == 'mobilenet':
    data_shape = [1, 3, 224, 224]
    execute(mobilenet.MobileNetV2, parser.model, data_shape, try_use_gpu)
elif parser.model == 'c3d':
    data_shape = [1, 3, 12, 112, 112]
    execute(c3d.C3D, parser.model, data_shape, try_use_gpu)
elif parser.model == 'tiny-yolo':
    data_shape = [1, 3, 224, 224]
    execute(lambda: yolo3.Darknet('config/tiny-yolo.cfg'), parser.model, data_shape, try_use_gpu)
elif parser.model == 'yolo3':
    data_shape = [1, 3, 224, 224]
    execute(lambda: yolo3.Darknet('config/yolo3.cfg'), parser.model, data_shape, try_use_gpu)
elif parser.model == 'cifarnet':
    data_shape = [1, 3, 32, 32]
    execute(cifarnet.CifarNet, parser.model, data_shape, try_use_gpu)
elif parser.model == 'deepspeech':
    data_shape = [1, 3, 224, 224]
    execute(deepspeech.DeepSpeech, parser.model, data_shape, try_use_gpu)
elif parser.model == 'resnet50-ssd':
    data_shape = [1, 3, 300, 300]
    execute(lambda: resnet50_ssd.SSD300(50), parser.model, data_shape, try_use_gpu)
else:
    raise Exception('Model %s is not defined.' % parser.model)
