

from torch.nn.modules.utils import  _pair
import torch.nn.modules.conv as Mconv
import torch.nn.functional as F
import torch.nn as nn

class partialConv2d(Mconv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(partialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.weightShape = self.weight.shape
    def forward(self, input):
        for i in range(0,input.shape[1]):
            if i == 0:
                out = F.conv2d(input[:,i,:,:].unsqueeze_(1), self.weight[:,i,:,:].view(self.weightShape[0],1,self.weightShape[2],self.weightShape[3]), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
            else:
                out += F.conv2d(input[:,i,:,:].unsqueeze_(1), self.weight[:,i,:,:].view(self.weightShape[0],1,self.weightShape[2],self.weightShape[3]), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)



        return out


# class partialConv2d():
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(partialConv2d, self).__init__()
#         self.conv = []
#         for i in range(0,in_channels):
#             self.conv.append(nn.Conv2d(in_channels, 1, kernel_size, stride,
#                  padding, dilation, groups, bias))
#
#     def forward(self, input):
#         out = tensor([])
#         for i in range(0,input.shape[1]):
#             out += self.conv[i](input[:,i,:,:].unsqueeze_(1))
#
#         return out