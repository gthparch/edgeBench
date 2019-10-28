import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import math

__all__ = ['ResNetBase', 'resnet_base']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ConvBnReluLayer(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, stride, bias=False):
        super(ConvBnReluLayer, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ExtraLayers(nn.Module):

    def __init__(self, inplanes):
        super(ExtraLayers, self).__init__()
        self.convbnrelu1_1 = ConvBnReluLayer(inplanes, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu1_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.convbnrelu2_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu2_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.convbnrelu3_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu3_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)

    def forward(self, x):
        out1_1 = self.convbnrelu1_1(x)
        out1_2 = self.convbnrelu1_2(out1_1)
        out2_1 = self.convbnrelu2_1(out1_2)
        out2_2 = self.convbnrelu2_2(out2_1)
        out3_1 = self.convbnrelu3_1(out2_2)
        out3_2 = self.convbnrelu3_2(out3_1)
        out = self.avgpool(out3_2)
        return out1_2, out2_2, out3_2, out


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):

    def __init__(self, block, layers, width=1, num_classes=1000):
        self.inplanes = 64
        widths = [int(round(ch * width)) for ch in [64, 128, 256, 512]]
        super(ResNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        # change stride = 2, dilation = 1 in ResNet to stride = 1, dilation = 2 for the final _make_layer
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=1, dilation=2)
        # remove the final avgpool and fc layers
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(widths[3] * block.expansion, num_classes)
        # add extra layers
        self.extra_layers = ExtraLayers(self.inplanes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out38x38 = x
        x = self.layer3(x)
        x = self.layer4(x)
        out19x19 = x

        out10x10, out5x5, out3x3, out1x1 = self.extra_layers(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return out38x38, out19x19, out10x10, out5x5, out3x3, out1x1


def resnet_base(depth, width=1, pretrained=False, **kwargs):
    if (depth not in [18, 34, 50, 101, 152]):
        raise ValueError('Choose 18, 34, 50, 101 or 152 for depth')
    if ((width != 1) and pretrained):
        raise ValueError('Does not support pretrained models with width>1')

    name_dict = {18: 'resnet18', 34: 'resnet34', 50: 'resnet50', 101: 'resnet101', 152: 'resnet152'}
    layers_dict = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                   101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    block_dict = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
    model = ResNetBase(block_dict[depth], layers_dict[depth], width, **kwargs)
    if ((width == 1) and pretrained):
        model.load_state_dict(model_zoo.load_url(model_urls[name_dict[depth]]))
    return model


class L2Norm2d(nn.Module):
    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)


class MultiBoxLayer(nn.Module):

    def __init__(self, num_classes, num_anchors, in_planes):
        super(MultiBoxLayer, self).__init__()
        self.num_anchors = num_anchors
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(in_planes)):
            self.loc_layers.append(nn.Conv2d(in_planes[i], num_anchors[i] * 4, kernel_size=3, padding=1))
            self.cls_layers.append(
                nn.Conv2d(in_planes[i], num_anchors[i] * (num_classes + 1), kernel_size=3, padding=1))

    def forward(self, fms):
        loc_preds = []
        cls_preds = []
        for loc_layer, cls_layer, fm in zip(self.loc_layers, self.cls_layers, fms):
            loc_pred = loc_layer(fm)
            cls_pred = cls_layer(fm)
            # [N, #anchors*4,H,W] -> [N,H,W, #anchors*4] -> [N,H*W*#anchors, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            # [N,#anchors*(#classes+1),H,W] -> [N,H,W,#anchors*(#classes+1)] -> [N,H*W*#anchors,#classes+1]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, (self.num_classes + 1))
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)


class SSD300(nn.Module):
    input_size = 300
    num_anchors = [4, 6, 6, 6, 4, 4]
    in_planes = [512, 1024, 512, 256, 256, 256]

    def __init__(self, depth, width=1, num_classes=20):
        super(SSD300, self).__init__()
        self.base_network = resnet_base(depth, width)
        self.norm1 = L2Norm2d(20)
        self.num_classes = num_classes
        self.in_planes = [int(round(in_plane * width)) if i < 2 else in_plane \
                          for i, in_plane in enumerate(self.in_planes)]

    def forward(self, x):
        fms = self.base_network(x)
        fms[0] = self.norm1(fms[0])
        loc_preds, conf_preds = self.multibox(fms, self.num_classes, self.num_anchors, self.in_planes)
        return loc_preds, conf_preds

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
