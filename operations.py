import torch.nn as nn
import torch
import numpy as np
import scipy.optimize as opt
import math



class ReLuPCA(nn.Module):
    def __init__(self, args,ch):
        super(ReLuPCA, self).__init__()
        self.eigenVar = args.EigenVar
        self.microBlockSz = args.MicroBlockSz
        self.channels = ch
        self.clipType = args.clipType
        self.project = args.project
        self.projType = args.projType
        self.actBitwidth = args.actBitwidth
        self.perChQ = args.perCh
        if args.perCh:
            clampSize = ch
        else:
            clampSize = 1
        self.clampVal = torch.zeros(clampSize).cuda() # torch.zeros(self.featureSz)
        self.clampValLap = torch.zeros(clampSize).cuda() # torch.zeros(self.featureSz)
        self.clampValGaus  = torch.zeros(clampSize).cuda() # torch.zeros(self.featureSz)
        self.lapB = torch.zeros(clampSize).cuda() # torch.zeros(self.featureSz) #b of laplace distribution
        self.gauStd = torch.zeros(clampSize).cuda()
        self.numElems = torch.zeros(clampSize).cuda() # torch.zeros(self.featureSz)

        self.collectStats = False

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        if self.eigenVar == 1.0:
            assert (input.shape[1] == self.channels)
        input = self.relu(input)
        if self.project:
            rows = int(input.shape[2] / self.microBlockSz)
            columns = int(input.shape[3] / self.microBlockSz)
            # divide to micro blocks
            # for i in range(rows):
            #     for j in range(columns):
            #         #TODO - do concatenation in better way
            #         if (j==0) and (i==0):
            #             im = input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
            #                  j * self.microBlockSz:(j + 1) * self.microBlockSz].contiguous().view(input.shape[0],-1).t()
            #         else:
            #             im = torch.cat((im,input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
            #                  j * self.microBlockSz:(j + 1) * self.microBlockSz].contiguous().view(input.shape[0],-1).t()),dim=1)

            im = input.permute(0, 2, 3, 1).contiguous().view(-1, input.shape[1]).t()

            mn = torch.mean(im, dim=1, keepdim=True)
            # Centering the data
            im = (im - mn)

            #Calculate projection matrix if needed
            if self.collectStats:
                if self.projType == 'pca':
                    #covariance matrix
                    cov = torch.matmul(im,im.t()) /im.shape[1]
                    # svd
                    self.u, self.s, v = torch.svd(cov)
                    # self.u = self.u.clone().detach()
                if self.projType == 'eye':
                    self.u = torch.eye(im.shape[0]).cuda()
                    self.s = torch.ones(im.shape[0]).cuda()


            #projection
            imProj = torch.matmul(self.u.t(), im)

            if self.projType == 'pca' and self.eigenVar < 1.0:
                if self.collectStats:
                     #remove part of new base
                    # find index where eigenvalues are more important
                    sRatio = torch.cumsum(self.s, 0) / torch.sum(self.s)
                    self.cutIdx = (sRatio >= self.eigenVar).nonzero()[0]
                    # throw unimportant eigenvector
                    self.u = self.u[:, :self.cutIdx]
                    self.s = self.s[:self.cutIdx]
                    self.channels = self.cutIdx
                    imProj = imProj[:self.cutIdx, :]
                else:
                    imProj = imProj[:self.cutIdx, :]

            #make it to have std = 1
            normStd = torch.sqrt(self.s).unsqueeze(1)
            # imProj = imProj / normStd

            imProjQ = imProj.clone()


            if self.collectStats:
                #collect b of laplacian distribution
                if self.perChQ:
                    for i in range(0,self.channels):
                        self.lapB[i] += torch.sum(torch.abs(imProj[i,:]))
                        self.gauStd[i] += torch.sum((imProj[i,:]) ** 2)
                        self.numElems[i] += (imProj.shape[1])
                else:
                    self.lapB += torch.sum(torch.abs(imProj))
                    self.gauStd += torch.sum((imProj)**2)
                    self.numElems += (imProj.shape[1]*imProj.shape[0])
            else:
                # quantize and send to memory
                if self.perChQ:
                    for i in range(0,self.channels):
                        clampMax = self.clampVal[i].item()
                        clampMin = -1 * self.clampVal[i].item()
                        imProjQ[i,:] = torch.clamp(imProj[i, :], max=clampMax, min=clampMin)
                        dynMax = torch.max(imProjQ[i,:])
                        dynMin = torch.min(imProjQ[i,:])
                        imProjQ[i,:] = act_quant(imProjQ[i,:], max = dynMax, min = dynMin, bitwidth =  self.actBitwidth)

                else:
                    clampMax = self.clampVal.item()
                    clampMin = -1 * self.clampVal.item()
                    imProjQ = torch.clamp(imProj, max=clampMax, min=clampMin)
                    dynMax = torch.max(imProjQ)
                    dynMin = torch.min(imProjQ)
                    imProjQ = act_quant(imProjQ, max=dynMax, min=dynMin,
                                              bitwidth= self.actBitwidth)

            # read from memory , project back and centering back
            # imProjQ = (torch.matmul(self.u, imProjQ * normStd) + mn).t()
            imProjQ = (torch.matmul(self.u, imProjQ ) + mn).t()

            out = imProjQ.view(input.shape[0], input.shape[2], input.shape[3], input.shape[1]).permute(0, 3, 1, 2)
         #   input = imProjQ.view(input.shape[0], input.shape[2], input.shape[3], input.shape[1]).permute(0, 3, 1, 2)

            snr = torch.sum((out - input)**2) / torch.numel(input)
#            print(snr)
            input = out
            # TODO - ugly code
            # for i in range(rows):
            #     for j in range(columns):
            #         currIdx = i*rows + j
            #         imProjQCurr = imProj[currIdx*input.shape[0]:(currIdx+1)*input.shape[0],:]
            #         imProjQCurr = imProjQCurr.view_as(input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
            #              j * self.microBlockSz:(j + 1) * self.microBlockSz])
            #         input[:, :, i * self.microBlockSz:(i + 1) * self.microBlockSz,
            #         j * self.microBlockSz:(j + 1) * self.microBlockSz] = imProjQCurr

        return input

    def updateClamp(self):
        if self.clipType == 'laplace':
            self.clampVal = self.clampValLap
        else:
            self.clampVal = self.clampValGaus

    def updateClampValLap(self):
        if self.perChQ:
            for i in range(0,self.channels):
                self.clampValLap[i] = opt.minimize_scalar(
                    lambda x: mse_laplace(x, b=(self.lapB[i] / self.numElems[i]).item(), num_bits=self.actBitwidth)).x

        else:
            self.clampValLap = opt.minimize_scalar(
                lambda x: mse_laplace(x, b=(self.lapB / self.numElems).item(), num_bits=self.actBitwidth)).x

    def updateClampValGaus(self):
        if self.perChQ:
                for i in range(0, self.channels):
                    self.clampValGaus[i] = opt.minimize_scalar(
                        lambda x: mse_gaussian(x, sigma=torch.sqrt((self.gauStd[i] / (self.numElems[i]-1))).item(),
                                              num_bits=self.actBitwidth)).x

        else:
                self.clampValGaus = opt.minimize_scalar(
                    lambda x: mse_gaussian(x, sigma=torch.sqrt((self.gauStd / (self.numElems - 1))).item(), num_bits=self.actBitwidth)).x


#taken from https://github.com/submission2019/cnn-quantization/blob/master/optimal_alpha.ipynb
def mse_laplace(alpha, b, num_bits):
    return 2 * (b ** 2) * np.exp(-alpha / b) + ((alpha ** 2) / (3 * 2 ** (2 * num_bits)))

def mse_gaussian(alpha, sigma, num_bits):
    clipping_err = (sigma ** 2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma * np.sqrt(2.0)))) - \
                   np.sqrt(2.0 / np.pi) * alpha * sigma * (np.e ** ((-1) * (0.5 * (alpha ** 2)) / sigma ** 2))
    quant_err = (alpha ** 2) / (3 * (2 ** (2 * num_bits)))
    return clipping_err + quant_err

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
