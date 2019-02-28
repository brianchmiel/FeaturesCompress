
from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F
import torch


class CompressLoss(Module):
    def __init__(self, args):
        super(CompressLoss, self).__init__()
        self.lmbda = args.lmbda
        self.crossEntropyLoss = CrossEntropyLoss().cuda()

    def forward(self, input, target, modelSnr):
        crossEntropyLoss = self.crossEntropyLoss(input, target)
        compressLoss = self.lmbda * (1/modelSnr)
        totalLoss = crossEntropyLoss + compressLoss
        return totalLoss, crossEntropyLoss, compressLoss

#
# class EntropyEstimation(Module):
#     def __init__(self,args):
#         super(EntropyEstimation, self).__init()
#
#
#     def forward(self, input, bits):
#         bins = linspace(3, 10, steps=5)
#
#         input = input.unsqueeze(2).repeat(1,1,bins.shape[0])
#
#
#         list(map(lambda x,y,z: (F.relu(x-z) - F.relu(x-y)), input ,bins,torch.cat(torch.zeros(1), bins[:-1])))
