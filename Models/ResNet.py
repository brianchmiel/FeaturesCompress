import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from absorbe_bn import search_absorbe_bn
from operations import ReLuPCA

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def flatten(x):
    return x.view(x.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = ReLuPCA(args, planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = ReLuPCA(args, planes)

        self.downsample = downsample

        self.stride = stride

    def forward(self, x):

        residue = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residue = self.downsample(x)

        out += residue
        out = self.relu2(out)
        return out

    def getLayers(self):
        layers = []
        for l in self._modules.values():
            if isinstance(l, nn.Sequential):
                for ls in l._modules.values():
                    layers.append(ls)
            else:
                layers.append(l)
        return layers


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = ReLuPCA(args, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = ReLuPCA(args, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = ReLuPCA(args, planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

    def getLayers(self):
        layers = []
        for l in self._modules.values():
            if isinstance(l, nn.Sequential):
                for ls in l._modules.values():
                    layers.append(ls)
            else:
                layers.append(l)
        return layers


class ResNetImagenet(nn.Module):

    def __init__(self, block, layers, args, zero_init_residual=False):
        super(ResNetImagenet, self).__init__()
        num_classes = args.nClasses
        self.name = args.model
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = ReLuPCA(args, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(args, block, 64, layers[0])
        self.layer2 = self._make_layer(args, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(args, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(args, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.statsState = False
        self.layersList = self.buildLayersList()
        self.ReLuPcaList = self.buildReluPcaList()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, args, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(args, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(args, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def loadPreTrained(self):
        self.load_state_dict(model_zoo.load_url(model_urls[self.name]), False)

    def buildLayersList(self):
        layersList = []
        for m in self._modules.values():
            if isinstance(m, nn.Sequential):
                for ms in m._modules.values():
                    layersList.extend(ms.getLayers())
            else:

                layersList.append(m)

        return layersList

    def buildReluPcaList(self):
        list = []
        for l in self.layersList:
            if isinstance(l, ReLuPCA):
                list.append(l)
        return list

    def enableStatisticPhase(self):
        self.statsState = True
        for l in self.ReLuPcaList:
            l.collectStats = True

    def disableStatisticPhase(self):
        self.statsState = False
        for l in self.ReLuPcaList:
            l.collectStats = False
            l.updateClampValLap()
            l.updateClampValGaus()
            l.updateClamp()

        # We quantize first and last layer to 8 bits
        self.ReLuPcaList[0].actBitwidth = 8
        self.ReLuPcaList[-1].actBitwidth = 8


class ResNetCifar(nn.Module):

    def __init__(self, depth, args):
        super(ResNetCifar, self).__init__()
        num_classes = args.nClasses
        assert (depth - 2) % 6 == 0, 'Depth should be 6n + 2'
        n = (depth - 2) // 6
        self.name = args.model
        self.dataset = args.dataset
        block = BasicBlock
        self.inplanes = 64
        fmaps = [64, 128, 256]  # CIFAR10

        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = ReLuPCA(args, 64)

        self.layer1 = self._make_layer(args, block, fmaps[0], n, stride=1)
        self.layer2 = self._make_layer(args, block, fmaps[1], n, stride=2)
        self.layer3 = self._make_layer(args, block, fmaps[2], n, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten = flatten
        self.fc = nn.Linear(fmaps[2] * block.expansion, num_classes)

        self.statsState = False
        self.layersList = self.buildLayersList()
        self.ReLuPcaList = self.buildReluPcaList()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, args, block, planes, blocks, stride=1):
        ''' Between layers convolve input to match dimensions -> stride = 2 '''

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(args, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(args, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, ):

        x = self.relu(self.bn(self.conv(x)))  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)  # 1x1
        x = self.flatten(x)  # Flatten
        x = self.fc(x)  # Dense
        return x

    def loadPreTrained(self):
        preTrainedDir = './preTrained/' + self.name + '/' + self.dataset + '/ckpt.t7'
        checkpoint = torch.load(preTrainedDir)
        self.load_state_dict(checkpoint['net'])

    def buildLayersList(self):
        layersList = []
        for m in self._modules.values():
            if isinstance(m, nn.Sequential):
                for ms in m._modules.values():
                    layersList.extend(ms.getLayers())
            else:

                layersList.append(m)

        return layersList

    def buildReluPcaList(self):
        list = []
        for l in self.layersList:
            if isinstance(l, ReLuPCA):
                list.append(l)
        return list

    def enableStatisticPhase(self):
        self.statsState = True
        for l in self.ReLuPcaList:
            l.collectStats = True

    def disableStatisticPhase(self):
        self.statsState = False
        for l in self.ReLuPcaList:
            l.collectStats = False
            l.updateClampVal()

        # We quantize first and last layer to 8 bits
        self.ReLuPcaList[0].actBitwidth = 8
        self.ReLuPcaList[-1].actBitwidth = 8


# =========== Cifar ResNet =============

def ResNet20(args):
    return ResNetCifar(depth=20, args=args)


def ResNet56(args):
    return ResNetCifar(depth=56, args=args)


# =========== Imagenet ResNet =============

def ResNet18(args):
    model = ResNetImagenet(BasicBlock, [2, 2, 2, 2], args)
    return model


def ResNet50(args):
    model = ResNetImagenet(Bottleneck, [3, 4, 6, 3], args)
    return model


def ResNet152(args):
    model = ResNetImagenet(Bottleneck, [3, 8, 36, 3], args)
    return model
