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
import scipy.optimize as opt






class pcaWhitening(nn.Module):
    def __init__(self, args, project, channels):
        super(pcaWhitening, self).__init__()
        self.eigenVar = args.EigenVar
        self.microBlockSz = args.MicroBlockSz
        assert (self.eigenVar >= 0 and self.eigenVar <= 1)  # ratio must be between 0 to 1
        self.project = project
        self.projType = args.ProjType
        self.actBitwidth = args.actBitwidth
        self.channels = channels
        self.featureSz = channels * self.microBlockSz * self.microBlockSz
        self.snr = torch.zeros(1).cuda()
        self.clampVal = torch.zeros(1) # torch.zeros(self.featureSz)
        self.lapB = torch.zeros(1) # torch.zeros(self.featureSz) #b of laplace distribution
        self.numElems = torch.zeros(1) # torch.zeros(self.featureSz)
        self.collectStats = False
    def forward(self, input):
        if self.project:
            rows = int(input.shape[2] / self.microBlockSz)
            columns = int(input.shape[3] / self.microBlockSz)
            # self.snr = 0
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
            mn = torch.mean(im, dim=1, keepdim=True)
            im = (im - mn)

            assert (im.shape[0] == self.featureSz)
            #Calculate projection matrix if needed
            if self.projType == 'pca':

                #covariance matrix
                cov = torch.matmul(im,im.t())  /im.shape[1]
                # svd
                u, s, v = torch.svd(cov)
                # u = vr_pca.vr_pca(im, 5, 1e-3)
                u=u.clone().detach()

            if self.projType == 'eye':
                u = torch.eye(im.shape[0]).cuda()

            #projection
            imProj = torch.matmul(u.t(), im)

            #remove part of new base
            if self.projType == 'pca' and self.eigenVar < 1.0:
                # find index where eigenvalues are more important
                sRatio = torch.cumsum(s, 0) / torch.sum(s)
                cutIdx = (sRatio >= self.eigenVar).nonzero()[0]
                # throw unimportant eigenvector
                u = u[:, :cutIdx]
                imProj = imProj[:cutIdx,:]
            # self.nonZeroElem += imProj.nonzero()[-1][0]

            #make it to have std = 1
            stdd = torch.sqrt(s).unsqueeze(1) # torch.ones( (s.shape[0],1)).cuda() #
            imProj = imProj / stdd
            # quantize and send to memory
            imProjQ = imProj.clone()

            # mxx = torch.max(imProj)
            # mnn = torch.min(imProj)
            # imProjQ = act_quant(torch.clamp(imProj, max=mxx, min=mnn), max=mxx, min=mnn,
            #                           bitwidth=self.actBitwidth)

            if self.collectStats:
                # for i in range(0, imProj.shape[0]):
                #     self.lapB[i] += torch.sum(torch.abs(imProj[i,:]))
                #     self.numElems[i] += imProj.shape[1]
                self.lapB += torch.sum(torch.abs(imProj))
                self.numElems += (imProj.shape[1]*imProj.shape[0])

            else:
                for i in range(0,imProj.shape[0]):
                    mxx = self.clampVal.item() # torch.max(imProj[i,:])
                    mnn = -1 * self.clampVal.item() # torch.min(imProj[i,:])
                    imProjQ[i,:] = act_quant(torch.clamp(imProj[i,:],max=mxx,min=mnn), max = mxx, min = mnn, bitwidth = self.actBitwidth)

            # self.QnonZeroElem += imProjQ.nonzero()[-1][0]

            self.snr += torch.sum((imProj - imProjQ) ** 2) # / torch.numel(imProj)

            # read from memory , project back and centering back
            imProjQ = (torch.matmul(u, imProjQ * stdd) + mn).t()




            # TODO - ugly code
            for i in range(rows):
                for j in range(columns):
                    currIdx = i*rows + j
                    imProjQCurr = imProjQ[currIdx*input.shape[0]:(currIdx+1)*input.shape[0],:]
                    imProjQCurr = imProjQCurr.view_as(input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
                         j * self.microBlockSz:(j + 1) * self.microBlockSz])
                    input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
                    j * self.microBlockSz:(j + 1) * self.microBlockSz] = imProjQCurr



        return input

    def act_clamp(self, x, clamp_val):
        x = F.relu(x + torch.abs(clamp_val)) - F.relu(x - torch.abs(clamp_val))
        return x

    def updateClampVal(self):
        # for i in range(0,self.featureSz):
        #     self.clampVal[i] = opt.minimize_scalar(lambda x: mse_laplace(x, b=(self.lapB[i] /self.numElems[i]) , num_bits=self.actBitwidth)).x
        self.clampVal = opt.minimize_scalar(
            lambda x: mse_laplace(x, b=(self.lapB / self.numElems), num_bits=self.actBitwidth)).x

#taken from https://github.com/submission2019/cnn-quantization/blob/master/optimal_alpha.ipynb
def mse_laplace(alpha, b, num_bits):
    return 2 * (b ** 2) * np.exp(-alpha / b) + ((alpha ** 2) / (3 * 2 ** (2 * num_bits)))


def act_quant(x, max, min, bitwidth):
    act_scale = (2 ** bitwidth - 1) / (max - min)
    q_x = (Round.apply((x - min)* act_scale) * 1 / act_scale ) + min
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
