import math

import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn
from tqdm import trange


def optimal_matrix(cov):
    tmp_u, _, _ = torch.svd(cov)
    u = tmp_u.clone().requires_grad_()
    optimizer = torch.optim.SGD([u], lr=1e-5, momentum=0, weight_decay=0)
    alpha = 0.1
    beta = 100
    for _ in trange(10000):
        optimizer.zero_grad()
        d = torch.matmul(u.transpose(0, 1), torch.matmul(cov, u))
        s = torch.diag(d).sum()
        norm = torch.norm(u, p=1)
        orth = torch.norm(torch.matmul(u.transpose(0, 1), u) - torch.eye(u.shape[0]).to(u))
        # print(orth)
        loss = s + beta * orth + alpha * norm
        loss.backward()
        optimizer.step()
    s = torch.diag(torch.matmul(u.transpose(0, 1), torch.matmul(cov, u)))
    # DEBUG
    print(torch.norm(tmp_u - u).item(), torch.nonzero(u < 1e-5).size(0), torch.nonzero(tmp_u < 1e-5).size(0))
    return u.clone(), s.clone()  # TODO


def get_projection_matrix(im, projType):
    if projType == 'pca':
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        # svd
        u, s, _ = torch.svd(cov)
    elif projType == 'eye':
        u, s = torch.eye(im.shape[0]), torch.ones(im.shape[0])
    elif projType == 'optim':
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        # do optimization
        u, s = optimal_matrix(cov)
    else:
        raise ValueError("Wrong projection type")
    return u, s


class ReLuPCA(nn.Module):
    def __init__(self, args, ch):
        super(ReLuPCA, self).__init__()
        self.eigenVar = args.EigenVar
        self.microBlockSz = args.MicroBlockSz
        self.channels = ch
        self.clipType = args.clipType
        self.project = args.project
        self.projType = args.projType
        self.actBitwidth = args.actBitwidth
        self.perChQ = args.perCh
        clampSize = ch if self.perChQ else 1

        self.clampVal = torch.zeros(clampSize)
        self.clampValLap = torch.zeros(clampSize)
        self.clampValGaus = torch.zeros(clampSize)
        self.lapB = torch.zeros(clampSize)  # b of laplace distribution
        self.gauStd = torch.zeros(clampSize)
        self.numElems = torch.zeros(clampSize)

        self.collectStats = False

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        if self.eigenVar == 1.0:
            assert (input.shape[1] == self.channels)
        input = self.relu(input)

        if self.project:
            N, C, H, W = input.shape  # N x C x H x W
            im = input.detach().transpose(0, 1).contiguous()  # C x N x H x W
            im = im.view(im.shape[0], -1)  # C x (NxHxW)

            mn = torch.mean(im, dim=1, keepdim=True)
            # Centering the data
            im = (im - mn)

            # Calculate projection matrix if needed
            if self.collectStats:
                self.u, self.s = get_projection_matrix(im, self.projType)

            # projection
            imProj = torch.matmul(self.u.t(), im)

            if self.projType == 'pca' and self.eigenVar < 1.0:
                if self.collectStats:
                    # remove part of new base
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

            # make it to have std = 1
            # normStd = torch.sqrt(self.s).unsqueeze(1)
            # imProj = imProj / normStd
            imProjQ = imProj.clone()
            if self.collectStats:
                # collect b of laplacian distribution
                if self.perChQ:
                    for i in range(0, self.channels):
                        self.lapB[i] += torch.sum(torch.abs(imProj[i, :]))
                        self.gauStd[i] += torch.sum((imProj[i, :]) ** 2)
                        self.numElems[i] += (imProj.shape[1])
                else:
                    self.lapB += torch.sum(torch.abs(imProj))
                    self.gauStd += torch.sum((imProj) ** 2)
                    self.numElems += (imProj.shape[1] * imProj.shape[0])
            else:
                # quantize and send to memory
                if self.perChQ:
                    for i in range(0, self.channels):
                        clampMax = self.clampVal[i].item()
                        clampMin = -1 * self.clampVal[i].item()
                        imProjQ[i, :] = torch.clamp(imProj[i, :], max=clampMax, min=clampMin)
                        dynMax = torch.max(imProjQ[i, :])
                        dynMin = torch.min(imProjQ[i, :])
                        imProjQ[i, :] = act_quant(imProjQ[i, :], max=dynMax, min=dynMin, bitwidth=self.actBitwidth)

                else:
                    clampMax = self.clampVal.item()
                    clampMin = -1 * self.clampVal.item()
                    imProjQ = torch.clamp(imProj, max=clampMax, min=clampMin)
                    dynMax = torch.max(imProjQ)
                    dynMin = torch.min(imProjQ)
                    imProjQ = act_quant(imProjQ, max=dynMax, min=dynMin,
                                        bitwidth=self.actBitwidth)

            # read from memory, project back and centering back
            # imProjQ = (torch.matmul(self.u, imProjQ * normStd) + mn).t()
            imProjQ = torch.matmul(self.u, imProjQ)

            # Bias Correction
            imProjQ = (imProjQ - torch.mean(imProjQ, dim=1, keepdim=True))

            # return original mean
            imProjQ = (imProjQ + mn)

            input = imProjQ.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W

        return input

    def updateClamp(self):
        if self.clipType == 'laplace':
            self.clampVal = self.clampValLap
        else:
            self.clampVal = self.clampValGaus

    def updateClampValLap(self):
        if self.perChQ:
            for i in range(0, self.channels):
                self.lapB[i] = (self.lapB[i] / self.numElems[i])
                if self.lapB[i] > 0:
                    self.clampValLap[i] = opt.minimize_scalar(
                        lambda x: mse_laplace(x, b=self.lapB[i].item(), num_bits=self.actBitwidth)).x
                else:
                    self.clampValLap[i] = 0

        else:
            self.lapB = (self.lapB / self.numElems)
            self.clampValLap = opt.minimize_scalar(
                lambda x: mse_laplace(x, b=self.lapB.item(), num_bits=self.actBitwidth)).x

    def updateClampValGaus(self):
        if self.perChQ:
            for i in range(0, self.channels):
                self.gauStd[i] = torch.sqrt((self.gauStd[i]) / (self.numElems[i] - 1))
                if self.gauStd[i] > 0:
                    self.clampValGaus[i] = opt.minimize_scalar(
                        lambda x: mse_gaussian(x, sigma=self.gauStd[i].item(),
                                               num_bits=self.actBitwidth)).x
                else:
                    self.clampValGaus[i] = 0
        else:
            self.gauStd = torch.sqrt((self.gauStd) / (self.numElems - 1))
            self.clampValGaus = opt.minimize_scalar(
                lambda x: mse_gaussian(x, sigma=self.gauStd.item(), num_bits=self.actBitwidth)).x


# taken from https://github.com/submission2019/cnn-quantization/blob/master/optimal_alpha.ipynb
def mse_laplace(alpha, b, num_bits):
    #  return 2 * (b ** 2) * np.exp(-alpha / b) + ((alpha ** 2) / (3 * 2 ** (2 * num_bits)))
    exp_val = 1e300 if -alpha / b > 690 else np.exp(-alpha / b)  # prevent overflow
    res = (b ** 2) * exp_val + ((alpha ** 2) / (24 * 2 ** (2 * num_bits)))
    return res


def mse_gaussian(alpha, sigma, num_bits):
    clipping_err = (sigma ** 2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma * np.sqrt(2.0)))) - \
                   np.sqrt(2.0 / np.pi) * alpha * sigma * (np.e ** ((-1) * (0.5 * (alpha ** 2)) / sigma ** 2))
    quant_err = (alpha ** 2) / (3 * (2 ** (2 * num_bits)))
    return clipping_err + quant_err


def act_quant(x, max, min, bitwidth):
    if max != min:
        act_scale = (2 ** bitwidth - 1) / (max - min)
        q_x = (Round.apply((x - min) * act_scale) * 1 / act_scale) + min
    else:
        q_x = x
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
