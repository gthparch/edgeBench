import tensorflow as tf

import argparse
from timeit import default_timer as timer

import numpy as np

from models import inception, resnet, yolo, alexnet

args = argparse.ArgumentParser()
args.add_argument('--model', type=str, help='Model name.')
args.add_argument('-i', '--iteration', default=1000, type=int, help='Test iteration.')
parser = args.parse_args()


def execute(model_f, name, data_shape):
    net = model_f()
    start = timer()
    for _ in range(parser.iteration):
        data = np.random.random_sample(data_shape)
        net.predict(data)
    print(f'{name} inference time (sec): {(timer() - start) / parser.iteration:.5f}')


if parser.model == 'resnet-50':
    data_shape = [1, 224, 224, 3]
    f = tf.keras.applications.ResNet50
    execute(f, parser.model, data_shape)
elif parser.model == 'resnet-18':
    data_shape = (224, 224, 3)
    execute(lambda: resnet.ResNet18(data_shape, 1000), parser.model, [1, 224, 224, 3])
elif parser.model == 'resnet-101':
    data_shape = (224, 224, 3)
    execute(lambda: resnet.ResNet101(data_shape, 1000), parser.model, [1, 224, 224, 3])
elif parser.model == 'alexnet':
    data_shape = (224, 224, 3)
    execute(lambda: alexnet.alexnet(), parser.model, [1, 224, 224, 3])
elif parser.model == 'vgg16':
    data_shape = [1, 224, 224, 3]
    f = tf.keras.applications.VGG16
    execute(f, parser.model, data_shape)
elif parser.model == 'vgg19':
    data_shape = [1, 224, 224, 3]
    f = tf.keras.applications.VGG19
    execute(f, parser.model, data_shape)
elif parser.model == 'xception':
    data_shape = [1, 299, 299, 3]
    f = tf.keras.applications.Xception
    execute(f, parser.model, data_shape)
elif parser.model == 'mobilenet':
    data_shape = [1, 224, 224, 3]
    f = tf.keras.applications.MobileNetV2
    execute(f, parser.model, data_shape)
elif parser.model == 'inception':
    execute(lambda: inception.create_model(), parser.model, [1, 299, 299, 3])
elif parser.model == 'tiny-yolo':
    execute(lambda: yolo.tiny_yolo(), parser.model, [1, 224, 224, 3])
else:
    raise Exception('Model %s is not defined.' % parser.model)
