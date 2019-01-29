import torch.nn.functional as F
import torch.nn as nn
import math
import torch
from torch.autograd import Variable
from scipy.linalg import hadamard
from torch.nn.parameter import Parameter
import time
from pca_only import pca
import numpy as np
import time


class Quant(nn.Module):
    def __init__(self, args):
        super(Quant, self).__init__()
        self.clamp_val = torch.zeros(1)
        self.actBitwidth = args.actBitwidth

    def forward(self,input,base):


        #project input on new base
        # projection
        imProj = torch.matmul(base.t(), im)
        #quantize
        c_x = self.act_clamp(imProj, self.clamp_val)
        x = act_quant(c_x, self.clamp_val, self.bitwidth)
        #return to original base

        return input




class projBase(nn.Module):
    def __init__(self, args):
        super(projBase, self).__init__()
        self.eigenVar = args.EigenVar
        self.microBlockSz = args.MicroBlockSz
        assert (self.eigenVar >= 0 and self.eigenVar <= 1)  # ratio must be between 0 to 1
        self.b = Parameter(torch.nn.init.normal(torch.ones(1)))

        self.snr = torch.zeros(1)




    def forward(self, input):
        rows = int(input.shape[2] / self.microBlockSz)
        columns = int(input.shape[3] / self.microBlockSz)
        self.snr = 0
        self.nonZeroElem = torch.zeros(1,dtype = torch.long).cuda()
        self.QnonZeroElem = torch.zeros(1, dtype = torch.long).cuda()
        tmp = 0
        # divide to micro blocks
        for i in range(rows):
            for j in range(columns):
                #TODO - do concatenation in better way
                if (j==0) and (i==0):
                    im = input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
                         j * self.microBlockSz:(j + 1) * self.microBlockSz].contiguous().view(input.shape[0],-1).t()
                else:
                    im = torch.cat((im,input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
                         j * self.microBlockSz:(j + 1) * self.microBlockSz].contiguous().view(input.shape[0],-1).t()),dim=1)

        # Centering the data
        mn = torch.mean(im,dim = 1,keepdim=True)
        im = (im - mn)
        #covariance matrix
        cov = torch.matmul(im,im.t())/im.shape[1]
        # svd
        u, s, v = torch.svd(cov)
        # u = vr_pca.vr_pca(im, 5, 1e-3)
        u=u.clone().detach()


        self.nonZeroElem += imProj.nonzero()[-1][0]
        # quantize and send to memory

        self.b.data = torch.abs(self.b)
        tmp1 = imProj / self.b
        tmp = Round.apply(tmp1)
        imProjQ = self.b * tmp

        self.QnonZeroElem += imProjQ.nonzero()[-1][0]
        # TODO - calculate entropy, for now we use SNR
        self.snr += torch.sum((imProj - imProjQ) ** 2) # / torch.numel(imProj)

        # read from memory and project back
        imProjQ = torch.matmul(u, imProjQ).t()


        # TODO - ugly code
        for i in range(rows):
            for j in range(columns):
                currIdx = i*rows + j
                imProjQCurr = imProjQ[currIdx*input.shape[0]:(currIdx+1)*input.shape[0],:]
                imProjQCurr = imProjQCurr.view_as(input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
                     j * self.microBlockSz:(j + 1) * self.microBlockSz])
                input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
                j * self.microBlockSz:(j + 1) * self.microBlockSz] = imProjQCurr

                    # #find index where eigenvalues are more important
                    # sRatio = torch.cumsum(s,0) / torch.sum(s)
                    # cutIdx = (sRatio >= self.eigenVar).nonzero()[0]
                    # #throw unimporatnt eigenvector
                    # u[:,cutIdx:] = 0
                    # #project to new space
                    # out[i,:,:,:].data = torch.matmul(u.t(),im)

        return input

    def act_clamp(self, x, clamp_val):
        x = F.relu(x + torch.abs(clamp_val)) - F.relu(x - torch.abs(clamp_val))
        return x

def act_quant(x, act_max_value, bitwidth):
    act_scale = (2 ** bitwidth - 1) / act_max_value
    q_x = Round.apply(x * act_scale) * 1 / act_scale
    return q_x

class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        round = (x).round()
        return round.to(x.device)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input, None, None
