
import torch
import torch.nn as nn
import torch.nn.functional as F
# from operations import partialConv2d
from collections import OrderedDict


class BasicBlockPartial(nn.Module):
    expansion = 1

    def __init__(self, args,  in_planes, planes, stride=1):
        super(BasicBlockPartial, self).__init__()
        self.conv1 = partialConv2d(args, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = partialConv2d(args, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                partialConv2d(args, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def getOps(self):
        ops = []
        for k,v in self._modules.items():
            if isinstance(v,nn.Sequential):
                for kk, vv in v._modules.items():
                    ops.append(vv)
            else:
                ops.append(v)
        return ops



class TinyResNetPartial(nn.Module):
    def __init__(self, args):
        super(TinyResNetPartial, self).__init__()
        self.in_planes = 128
        self.opsList = []
        self.conv1 = partialConv2d(args, 3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.opsList.append(self.conv1)
        self.bn1 = nn.BatchNorm2d(128)
        self.opsList.append(self.bn1)
        num_blocks = [2, 2]
        self.layer1, ops = self._make_layer(args, BasicBlockPartial, 128, num_blocks[0], stride=1)
        self.opsList.extend(ops)
        self.layer2, ops = self._make_layer(args, BasicBlockPartial, 256, num_blocks[1], stride=2)
        self.opsList.extend(ops)
        self.linear = nn.Linear(256*BasicBlockPartial.expansion, args.nClasses)
        self.opsList.append(self.linear)
    def _make_layer(self, args, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        ops = []
        for stride in strides:
            currBlock = block(args, self.in_planes, planes, stride)
            layers.append(currBlock)
            ops.extend(currBlock.getOps())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers), ops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def loadPreTrained(self,path):
        fullPath = path + '/ckpt.t7'
        checkpoint = torch.load(fullPath)
        self.load_state_dict(checkpoint['net'], False)
        #Build list that contain the key of state_dict that include convolutions (include in shortcut)
        convState = []
        for k in checkpoint['net']:
            if ('conv' in k or 'shortcut.0' in k) and 'weight' in k:
                prefix = k[:k.rindex('.')]
                convState.append(prefix)
        #load weights to partial conv
        idxConv = 0
        for ops in self.opsList:
            if isinstance(ops, partialConv2d):
                ops.loadPreTrained(w = checkpoint['net'].get(convState[idxConv] + '.weight'), b =checkpoint['net'].get(convState[idxConv] + '.bias'))
                idxConv+=1
        assert(idxConv == len(convState))









