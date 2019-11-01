import argparse
import numpy as np

from timeit import default_timer as timer

import caffe

args = argparse.ArgumentParser()
args.add_argument('--model', help='Model name')
args.add_argument('--cpu', help='Use cpu or not', default=False, type=bool)
args.add_argument('-i', '--iteration', default=1000, type=int, help='Test iteration.')
parser = args.parse_args()


def run(name, gpu):
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    if name == 'alexnet':
        path = 'models/alexnet.prototxt'
        shape = [1, 3, 224, 224]
    elif name == 'resnet-50':
        path = 'models/resnet50.prototxt'
        shape = [1, 3, 224, 224]
    elif name == 'resnet-101':
        path = 'models/resnet101.prototxt'
        shape = [1, 3, 224, 224]
    elif name == 'vgg16':
        path = 'models/vgg16.prototxt'
        shape = [1, 3, 224, 224]
    elif name == 'vgg19':
        path = 'models/vgg19.prototxt'
        shape = [1, 3, 224, 224]
    elif name == 'mobilenet':
        path = 'models/mobilenet.prototxt'
        shape = [1, 3, 224, 224]
    elif name == 'inception':
        path = 'models/inception.prototxt'
        shape = [1, 3, 299, 299]
    elif name == 'xception':
        path = 'models/xception.prototxt'
        shape = [1, 3, 299, 299]
    else:
        raise Exception('Model not defined')

    net = caffe.Net(path, caffe.TEST)
    start = timer()
    for _ in range(parser.iteration):
        data = np.random.random_sample(shape)
        net.blobs['data'].data[...] = data
        net.forward()
    print(f'{name} inference time (sec): {(timer() - start) / parser.iteration:.5f}')


run(parser.model, not parser.cpu)
