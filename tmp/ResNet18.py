import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from operations import pcaWhitening
import torch.utils.model_zoo as model_zoo



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, in_planes, planes, stride=1,pcaList=[False,False]):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.pca1 = pcaWhitening(args, pcaList[0],planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.pca2 = pcaWhitening(args, pcaList[1],planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pca1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.pca2(out)
        return out

    def getLayers(self):
        layers = []
        for l in self._modules.values():
            if isinstance(l,nn.Sequential):
                for ls in l._modules.values():
                    layers.append(ls)
            else:
                layers.append(l)
        return layers



class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.layerPcaList = self._createPcaLayerList(args.layerPCA)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pca1 = pcaWhitening(args, self.layerPcaList[0], 64)
        num_blocks = [2, 2, 2, 2]

        self.layer1 = self._make_layer(args, BasicBlock, 64, num_blocks[0], stride=1, pcaList = self.layerPcaList[1:5])

        self.layer2 = self._make_layer(args, BasicBlock, 128, num_blocks[1], stride=2, pcaList = self.layerPcaList[5:9])
        self.layer3 = self._make_layer(args, BasicBlock, 256, num_blocks[2], stride=2, pcaList = self.layerPcaList[9:13])
        self.layer4 = self._make_layer(args, BasicBlock, 512, num_blocks[3], stride=2, pcaList = self.layerPcaList[13:17])

     #   self.pcaLayer = args.layerCompress

        self.linear = nn.Linear(512*BasicBlock.expansion, args.nClasses)
        self.statsState = False

        self.totalSnr = 0
        self.layersList = self.buildLayersList()


    def _createPcaLayerList(self, layerPca):
        lst = np.array([False for i in range(0,17)])
        if layerPca != None:
            idx = np.array(layerPca)
            lst[idx] = True
        return list(lst)

    def enableStatisticPhase(self):
        self.statsState = True
        for l in self.layersList:
            if isinstance(l,pcaWhitening):
                l.collectStats = True


    def disableStatisticPhase(self):
        self.statsState = False
        for l in self.layersList:
            if isinstance(l, pcaWhitening):
                l.collectStats = False
                l.updateClampVal()
                l.updateProjectBase()


    def _make_layer(self, args, block, planes, num_blocks, stride,pcaList):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(block(args, self.in_planes, planes, stride,pcaList[i*2:(i+1)*2]))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pca1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def loadPreTrained(self,path,args):
        if args.dataset == 'cifar10':
            fullPath = path + '/ckpt.t7'
            checkpoint = torch.load(fullPath)
            self.load_state_dict(checkpoint['net'],False)
        elif args.dataset == 'imagenet':
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']),False)

        else:
            raise NameError('only support cifar10 and imagenet')



    def buildLayersList(self):
        layersList = []
        for m in self._modules.values():
            if isinstance(m,nn.Sequential):
                for ms in m._modules.values():
                    layersList.extend(ms.getLayers())
            else:

                layersList.append(m)

        return layersList

    def calcSnr(self):
        self.totalSnr = torch.zeros(1).cuda()
        for l in self.layersList:
            if isinstance(l,pcaWhitening):
                self.totalSnr += l.snr
        return self.totalSnr

    def getQparameter(self):
        Qparameters = []
        for l in self.layersList:
            if isinstance(l,pcaWhitening):
                Qparameters.append(l.b.item())
        return Qparameters

    def getQuantInform(self):
        elems = []
        for l in self.layersList:
            if isinstance(l,pcaWhitening):
                nonQ = l.nonZeroElem
                Q = l.QnonZeroElem
                elems.append(str(Q.item()) + '/' + str(nonQ.item()))
        return elems


