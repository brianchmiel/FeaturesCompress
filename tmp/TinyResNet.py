import torch.nn as nn
import torch.nn.functional as F
from .ResNet18 import BasicBlock



class TinyResNet(nn.Module):
    def __init__(self, args):
        super(TinyResNet, self).__init__()
        self.in_planes = 128
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        num_blocks = [2, 2]
        self.layer1 = self._make_layer(BasicBlock, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 256, num_blocks[1], stride=2)
        self.linear = nn.Linear(256*BasicBlock.expansion, args.nClasses)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



