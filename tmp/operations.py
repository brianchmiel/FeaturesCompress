import torch.nn as nn
import torch
import numpy as np
import scipy.optimize as opt




class ReLuPCA(nn.Module):
    def __init__(self, args,ch):
        super(ReLuPCA, self).__init__()
        self.eigenVar = args.EigenVar
        self.microBlockSz = args.MicroBlockSz
        self.channels = ch
        self.project = args.project
        self.projType = args.projType
        self.actBitwidth = args.actBitwidth
        self.perChQ = args.perCh
        if args.perCh:
            clampSize = ch
        else:
            clampSize = 1
        self.clampVal = torch.zeros(clampSize) # torch.zeros(self.featureSz)
        self.lapB = torch.zeros(clampSize) # torch.zeros(self.featureSz) #b of laplace distribution
        self.numElems = torch.zeros(clampSize) # torch.zeros(self.featureSz)

        self.collectStats = False

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        assert input.shape[1] == self.channels
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
            # Centering the data

            mn = torch.mean(im, dim=1, keepdim=True)
            im = (im - mn)

            #Calculate projection matrix if needed
            if self.collectStats:
                if self.projType == 'pca':
                    #covariance matrix
                    cov = torch.matmul(im,im.t())  /im.shape[1]
                    # svd
                    self.u, self.s, v = torch.svd(cov)
                    # self.u = self.u.clone().detach()
                if self.projType == 'eye':
                    self.u = torch.eye(im.shape[0]).cuda()
                    self.s = torch.ones(im.shape[0]).cuda()

            #projection
            imProj = torch.matmul(self.u.t(), im)

            #remove part of new base
            if self.collectStats and self.projType == 'pca' and self.eigenVar < 1.0:
                # find index where eigenvalues are more important
                sRatio = torch.cumsum(self.s, 0) / torch.sum(self.s)
                cutIdx = (sRatio >= self.eigenVar).nonzero()[0]
                # throw unimportant eigenvector
                self.u = self.u[:, :cutIdx]
                imProj = imProj[:cutIdx,:]
                self.s = self.s[:cutIdx]


            #make it to have std = 1
            normStd = torch.sqrt(self.s).unsqueeze(1)
            imProj = imProj / normStd

            imProjQ = imProj.clone()

            if self.collectStats:
                #collect b of laplacian distribution
                if self.perChQ:
                    for i in range(0,im.shape[0]):
                        self.lapB[i] += torch.sum(torch.abs(imProj[i,:]))
                        self.numElems[i] += (imProj.shape[1])
                else:
                    self.lapB += torch.sum(torch.abs(imProj))
                    self.numElems += (imProj.shape[1]*imProj.shape[0])
            else:
                # quantize and send to memory
                if self.perChQ:
                    for i in range(0,im.shape[0]):
                        mxx = self.clampVal[i].item()
                        mnn = -1 * self.clampVal[i].item()
                        imProjQ[i,:] = act_quant(torch.clamp(imProj[i,:],max=mxx,min=mnn), max = mxx, min = mnn, bitwidth = self.actBitwidth)

                else:
                    mxx = self.clampVal.item()
                    mnn = -1 * self.clampVal.item()
                    imProjQ = act_quant(torch.clamp(imProj, max=mxx, min=mnn), max=mxx, min=mnn,
                                              bitwidth=self.actBitwidth)

            # read from memory , project back and centering back
            imProjQ = (torch.matmul(self.u, imProjQ * normStd) + mn).t()
            #imProjQ = (torch.matmul(self.u, imProjQ ) + mn).t()

            input = imProjQ.view(input.shape[0], input.shape[2], input.shape[3], input.shape[1]).permute(0, 3, 1, 2)
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



    def updateClampVal(self):
        if self.perChQ:
            for i in range(0,self.channels):
                self.clampVal[i] = opt.minimize_scalar(
                    lambda x: mse_laplace(x, b=(self.lapB[i] / self.numElems[i]), num_bits=self.actBitwidth)).x

        else:
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
