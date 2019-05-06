import math

import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA
from torch.nn import functional as F
from tqdm import trange, tqdm

from entropy import shannon_entropy
from huffman import huffman_encode

eps = 1e-6
reps = 1e-2  # empirical value


class BatchNorm2dAbsorbed(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2dAbsorbed, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        if hasattr(self, 'absorbed'):
            return input
        else:
            return super(BatchNorm2dAbsorbed, self).forward(input)


def small(x):
    return torch.abs(x) < reps * torch.max(torch.abs(x))


def quantize1d_kmeans(x, num_bits=8, n_jobs=-1):
    orig_shape = x.shape
    x = np.expand_dims(x.flatten(), -1)
    # init = np.expand_dims(np.linspace(x.min(), x.max(), 2**num_bits), -1)
    kmeans = KMeans(n_clusters=2 ** num_bits, random_state=0, n_jobs=n_jobs)
    x_kmeans = kmeans.fit_predict(x)
    q_kmeans = np.array([kmeans.cluster_centers_[i] for i in x_kmeans])
    return q_kmeans.reshape(orig_shape)


def optimal_matrix(cov):
    tmp_u, _, _ = torch.svd(cov)
    n = torch.norm(tmp_u, p=1)
    tmp_u = tmp_u / n  # normalize to prevent divergence and make distance meaningful
    us = tmp_u.size(0) * tmp_u.size(1)
    ztmpu = torch.nonzero(small(tmp_u)).size(0)

    with torch.set_grad_enabled(True):
        u = tmp_u.clone().requires_grad_()
        optimizer = torch.optim.SGD([u], lr=1e-5, momentum=0, weight_decay=0)
        alpha = 1e-5  # TODO
        beta = 100
        for _ in trange(10000):
            optimizer.zero_grad()
            d = torch.matmul(u.transpose(0, 1), torch.matmul(cov, u))
            s = torch.diag(d).sum()
            norm = torch.norm(u, p=1)
            orth = torch.norm(torch.matmul(u.transpose(0, 1), u) - torch.eye(u.shape[0]).to(u) / n ** 2)
            loss = beta * orth + s + alpha * norm
            assert not torch.isnan(loss)
            loss.backward()
            optimizer.step()
    s = torch.diag(torch.matmul(u.transpose(0, 1), torch.matmul(cov, u)))
    # DEBUG
    zu = torch.nonzero(small(u)).size(0)
    u[small(u)] = 0
    u = u * n  # renormalize
    tmp_u = tmp_u * n
    tqdm.write("Distance {:.4f}. "
               "Learned zeros: {}/{}({:.4f}). "
               "PCA zeros: {}/{}({:.4f}).".format(torch.norm(tmp_u - u).item(),
                                                  zu, us, zu / us,
                                                  ztmpu, us, ztmpu / us))
    return u.clone(), s.clone()  # TODO


def get_projection_matrix(im, projType, eigenVar):
    if projType == 'pca':
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        # svd
        u, s, _ = torch.svd(cov)

    elif projType == 'pcaT':
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        # svd
        u, s, _ = torch.svd(cov)
        # find index where eigenvalues are more important
        sRatio = torch.cumsum(s, 0) / torch.sum(s)
        cutIdx = (sRatio >= eigenVar).nonzero()[0]
        # throw unimportant eigenvector
        u = u[:, :cutIdx]
        s = s[:cutIdx]
    elif projType == 'pcaQ':
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        # svd
        u, s, _ = torch.svd(cov)
        # mnOr = torch.mean(u, dim = 1 , keepdim=True)
        u = torch.tensor(quantize1d_kmeans(u.cpu().detach().numpy(), num_bits=8)).cuda()
        # mnCurr = torch.mean(u , dim = 1, keepdim=True)
        # bias corr
        # u = u - mnCurr + mnOr
    elif projType == 'pcaSparse':

        transformer = SparsePCA(alpha=1)
        transformer.fit(im.t().cpu().numpy())

        u = torch.tensor(transformer.components_)
        s = torch.zeros(1)
    elif projType == 'eye':
        u, s = torch.eye(im.shape[0]).to(im), torch.ones(im.shape[0]).to(im)
    elif projType == 'optim':
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        # do optimization
        u, s = optimal_matrix(cov)
    else:
        raise ValueError("Wrong projection type")
    return u, s


class ReLuPCA(nn.Module):
    def __init__(self, args, mxRelu6=False, clamp='gaus'):
        super(ReLuPCA, self).__init__()
        self.actBitwidth = args.actBitwidth
        self.projType = args.projType
        self.per_channel = args.perCh
        if self.projType == 'eye':
            self.stats = 'all' if not self.per_channel else 'channel'
        else:
            self.stats = 'first' if not self.per_channel else 'channel'

        self.project = args.project
        self.collectStats = True
        self.bit_count = None

        self.microBlockSz = args.MicroBlockSz
        self.channelsDiv = args.channelsDiv
        self.eigenVar = args.eigenVar
        dataBasicSize = 7 if args.dataset == 'imagenet' else 2
        self.clampType = clamp
        if mxRelu6:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    #    if self.microBlockSz > 1:
    #        assert (self.microBlockSz % dataBasicSize == 0 )
    #   assert (self.channelsDiv % 2 == 0)

    def featuresReshape(self, input, N, C, H, W):
        # check input
        if (self.microBlockSz > H):
            self.microBlockSz = H
        if (self.channelsDiv > C):
            self.channelsDiv = C
        assert (C % self.channelsDiv == 0)
        Ct = C // self.channelsDiv
        featureSize = self.microBlockSz * self.microBlockSz * Ct
        #     assert(featureSize < 7000) #constant for SVD converge TODO - check correct value

        input = input.view(-1, Ct, H, W)  # N' x Ct x H x W
        input = input.permute(0, 2, 3, 1)  # N' x H x W x Ct
        input = input.contiguous().view(-1, self.microBlockSz, W, Ct).permute(0, 2, 1, 3)  # N'' x W x microBlockSz x Ct
        input = input.contiguous().view(-1, self.microBlockSz, self.microBlockSz, Ct).permute(0, 3, 2,
                                                                                              1)  # N''' x Ct x microBlockSz x microBlockSz

        return input.contiguous().view(-1, featureSize).t()

    def featuresReshapeBack(self, input, N, C, H, W):

        input = input.t()
        Ct = C // self.channelsDiv

        input = input.view(-1, Ct, self.microBlockSz, self.microBlockSz).permute(0, 3, 2,
                                                                                 1)  # N'''  x microBlockSz x microBlockSz x Ct
        input = input.contiguous().view(-1, H, self.microBlockSz, Ct).permute(0, 2, 1,
                                                                              3)  # N''  x microBlockSz x H x Ct
        input = input.contiguous().view(-1, H, W, Ct).permute(0, 3, 1, 2)  # N' x Ct x H x W X

        input = input.contiguous().view(N, C, H, W)  # N x C x H x W

        return input

    def forward(self, input):

        if self.project:
            #   input = self.relu(input)
            N, C, H, W = input.shape  # N x C x H x W
            #  im = input.detach().transpose(0, 1).contiguous()
            # im = input.contiguous().view(input.shape[0], -1)
            im = self.featuresReshape(input, N, C, H, W)

            self.channels = im.shape[0]
            if not hasattr(self, 'clampValL'):
                self.register_buffer('clampValL', torch.zeros(self.channels))
            if not hasattr(self, 'clampValG'):
                self.register_buffer('clampValG', torch.zeros(self.channels))
            if not hasattr(self, 'clampVal'):
                self.register_buffer('clampVal', torch.zeros(self.channels))
            if not hasattr(self, 'lapB'):
                self.register_buffer('lapB', torch.zeros(self.channels))
            if not hasattr(self, 'gausStd'):
                self.register_buffer('gausStd', torch.zeros(self.channels))
            if not hasattr(self, 'numElems'):
                self.register_buffer('numElems', torch.zeros(self.channels))

            mn = torch.mean(im, dim=1, keepdim=True)
            # Centering the data
            im = im - mn

            # Calculate projection matrix if needed
            if self.collectStats:
                self.u, self.s = get_projection_matrix(im, self.projType, self.eigenVar)
                self.original_channels = self.u.shape[0]
                # sRatio = torch.cumsum(self.s, 0) / torch.sum(self.s)
                # cutIdx = (sRatio >= 0.8).nonzero()[0].type(torch.float32)
                # print((cutIdx / self.s.shape[0]).item())

            self.channels = self.u.shape[1]

            # projection
            imProj = torch.matmul(self.u.t(), im)

            mult = torch.zeros(1).to(imProj)
            add = torch.zeros(1).to(imProj)

            if self.collectStats:
                # collect b of laplacian distribution
                if self.stats == 'channel':
                    for i in range(0, self.channels):
                        self.lapB[i] += torch.sum(torch.abs(imProj[i, :]))
                        self.numElems[i] += imProj.shape[1]
                        self.gausStd[i] = torch.std(imProj[i, :])
                elif self.stats == 'first':
                    self.lapB += torch.sum(torch.abs(imProj[0, :]))
                    self.numElems += imProj.shape[1]
                    self.gausStd += torch.std(imProj[0, :])
                elif self.stats == 'all':
                    self.lapB += torch.sum(torch.abs(imProj))
                    self.numElems += imProj.numel()
                    self.gausStd += torch.std(imProj.view(-1))
                else:
                    raise ValueError("Wrong stats type")
            #     self.updateClamp(im)
            #
            #
            # clampMax = self.clampVal.item()
            # clampMin = -clampMax
            # imProj = torch.clamp(imProj, max=clampMax, min=clampMin)

            dynMax = torch.max(imProj)
            dynMin = torch.min(imProj)

            if self.actBitwidth < 30:
                imProj, mult, add = part_quant(imProj, max=dynMax, min=dynMin,
                                               bitwidth=self.actBitwidth)

            # for i in range(0, self.channels):
            #     clampMax = self.clampVal[i].item()
            #     clampMin = -clampMax
            #     imProj[i, :] = torch.clamp(imProj[i, :], max=clampMax, min=clampMin)

            # if self.stats == 'first' or self.stats == 'all':
            #     dynMax = torch.max(imProj)
            #     dynMin = torch.min(imProj)
            # for i in range(0, self.channels):
            #     if self.stats == 'channel':
            #         dynMax = torch.max(imProj[i, :])
            #         dynMin = torch.min(imProj[i, :])
            #
            #     if self.actBitwidth < 30:
            #         imProj[i, :], mult[i], add[i] = part_quant(imProj[i, :], max=dynMax, min=dynMin,
            #                                                    bitwidth=self.actBitwidth)

            #  if self.collectStats and self.actBitwidth < 30:
            self.act_size = imProj.numel()
            self.bit_per_entry = shannon_entropy(imProj).item()
            self.bit_count = self.bit_per_entry * self.act_size
            if False:  # TODO
                self.bit_countH = huffman_encode(imProj)
                self.bit_per_entryH = self.bit_countH / self.act_size

            # if self.actBitwidth < 30:
            #     for i in range(0, self.channels):
            #         imProj[i, :] = imProj[i, :] * mult[i] + add[i]

            if self.actBitwidth < 30:
                imProj = imProj * mult + add

            imProj = torch.matmul(self.u, imProj)

            # Bias Correction
            imProj = imProj - torch.mean(imProj, dim=1, keepdim=True)

            self.mse = torch.sum((imProj - im) ** 2)
            # return original mean
            imProj = imProj + mn

            # return to general
            #   input = imProj.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
            #  input = imProj.contiguous().view(N, C, H, W)# N x C x H x W
            input = self.featuresReshapeBack(imProj, N, C, H, W)

            self.collectStats = False

        input = self.relu(input)
        return input

    def updateClamp(self, data):
        self.lapB[0] = (self.lapB[0] / self.numElems[0])
        self.clampValL[0] = opt.minimize_scalar(
            lambda x: mse_laplace(x, b=self.lapB[0].item(), num_bits=self.actBitwidth)).x

        self.clampVal = self.clampValL[0]

        # origDist = torch.histc(data.cpu())
        # origDist = origDist / torch.sum(origDist)
        #
        # clamps = np.linspace(0.1, 2, 40) * self.clampValL[0].item()
        #
        # mses = []
        # divr = []
        # entr = []
        #
        # for c in range(0, len(clamps)):
        #
        #     imProj = torch.matmul(self.u.t(), data)
        #     clampMax = clamps[c]
        #     clampMin = -clampMax
        #
        #     if clampMax < torch.max(imProj) or clampMin > torch.min(imProj):
        #         imProj = torch.clamp(imProj, max=clampMax, min=clampMin)
        #
        #         dynMax = torch.max(imProj)
        #         dynMin = torch.min(imProj)
        #         if self.actBitwidth < 30:
        #             imProj, mult, add = part_quant(imProj, max=dynMax, min=dynMin,
        #                                            bitwidth=self.actBitwidth)
        #
        #         act_size = imProj.numel()
        #         bit_per_entry = shannon_entropy(imProj).item()
        #         bit_count = bit_per_entry * act_size
        #         #    print(bit_per_entry)
        #
        #         if self.actBitwidth < 30:
        #             imProj = imProj * mult + add
        #
        #         imProj = torch.matmul(self.u, imProj)
        #
        #         # Bias Correction
        #         imProj = imProj - torch.mean(imProj, dim=1, keepdim=True)
        #
        #         currDist = torch.histc(imProj.cpu(), max=torch.max(data), min=torch.min(data))
        #         currDist = currDist / torch.sum(currDist)
        #
        #         currDiv = origDist * (-1) * torch.log(origDist / currDist)
        #         currDiv[origDist == 0] = 0
        #         currDiv[currDist == 0] = 0
        #         divr.append(torch.sum(currDiv))
        #         mses.append(torch.mean((imProj - data) ** 2))
        #         entr.append(bit_per_entry)
        #
        # divr = torch.tensor(divr)
        # mses = torch.tensor(mses)
        # entr = torch.tensor(entr)
        #
        # # print('*****')
        # # print(divr)
        # # print(mses)
        # # print(entr)
        #
        # # idxs = (divr == torch.max(divr)).nonzero()
        # # idx = torch.argmin(mses[idxs])
        #
        # idxs = (mses == torch.min(mses)).nonzero()
        # idx = torch.argmax(divr[idxs])
        #
        # #  print(idxs[idx])
        #
        # self.clampVal = torch.tensor(clamps[idxs[idx]])
        #


class ConvBNPCA(nn.Conv2d):
    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, clamp='gaus'):
        super(ConvBNPCA, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        self.actBitwidth = args.actBitwidth
        self.projType = args.projType
        self.per_channel = args.perCh
        if self.projType == 'eye':
            self.stats = 'all' if not self.per_channel else 'channel'
        else:
            self.stats = 'first' if not self.per_channel else 'channel'

        self.project = args.project
        self.collectStats = True
        self.bit_count = None

        self.microBlockSz = args.MicroBlockSz
        self.channelsDiv = args.channelsDiv
        self.eigenVar = args.eigenVar
        dataBasicSize = 7 if args.dataset == 'imagenet' else 2
        self.clampType = clamp

    def featuresReshape(self, input, N, C, H, W):
        # check input
        if (self.microBlockSz > H):
            self.microBlockSz = H
        if (self.channelsDiv > C):
            self.channelsDiv = C
        assert (C % self.channelsDiv == 0)
        Ct = C // self.channelsDiv
        featureSize = self.microBlockSz * self.microBlockSz * Ct
        #     assert(featureSize < 7000) #constant for SVD converge TODO - check correct value

        input = input.view(-1, Ct, H, W)  # N' x Ct x H x W
        input = input.permute(0, 2, 3, 1)  # N' x H x W x Ct
        input = input.contiguous().view(-1, self.microBlockSz, W, Ct).permute(0, 2, 1, 3)  # N'' x W x microBlockSz x Ct
        input = input.contiguous().view(-1, self.microBlockSz, self.microBlockSz, Ct).permute(0, 3, 2,
                                                                                              1)  # N''' x Ct x microBlockSz x microBlockSz

        return input.contiguous().view(-1, featureSize).t()

    def featuresReshapeBack(self, input, N, C, H, W):

        input = input.t()
        Ct = C // self.channelsDiv

        input = input.view(-1, Ct, self.microBlockSz, self.microBlockSz).permute(0, 3, 2, 1)  # N'''  x microBlockSz x microBlockSz x Ct
        input = input.contiguous().view(-1, H, self.microBlockSz, Ct).permute(0, 2, 1, 3)  # N''  x microBlockSz x H x Ct
        input = input.contiguous().view(-1, H, W, Ct).permute(0, 3, 1, 2)  # N' x Ct x H x W X

        input = input.contiguous().view(N, C, H, W)  # N x C x H x W

        return input

    def get_stats_params(self, im):
        self.u, self.s = get_projection_matrix(im, self.projType, self.eigenVar)
        self.original_channels = self.u.shape[0]

        # update weights
        self.weight.data = self.weight.data.transpose(0, 3).unsqueeze(4)
        self.weight.data = torch.matmul(self.u.t(), self.weight.data)
        self.weight.data = self.weight.data.squeeze(4).transpose(0, 3).contiguous()
        if self.bias is None:
            zeros = torch.Tensor(self.out_channels).zero_().type(self.weight.data.type())
            self.bias = nn.Parameter(zeros)
        self.bias.data -= self.mn.squeeze()
        self.bias.data = torch.matmul(self.u.t(), self.bias.data)

    def forward(self, input):
        if self.project:

            self.channels = self.out_channels

            if not hasattr(self, 'clampValL'):
                self.register_buffer('clampValL', torch.zeros(self.channels))
            if not hasattr(self, 'clampValG'):
                self.register_buffer('clampValG', torch.zeros(self.channels))
            if not hasattr(self, 'clampVal'):
                self.register_buffer('clampVal', torch.zeros(self.channels))
            if not hasattr(self, 'lapB'):
                self.register_buffer('lapB', torch.zeros(self.channels))
            if not hasattr(self, 'gausStd'):
                self.register_buffer('gausStd', torch.zeros(self.channels))
            if not hasattr(self, 'numElems'):
                self.register_buffer('numElems', torch.zeros(self.channels))
            if not hasattr(self, 'mn'):
                self.register_buffer('mn', torch.zeros(self.channels, 1))

            # Calculate projection matrix if needed
            if self.collectStats:
                im = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
                N, C, H, W = im.shape  # N x C x H x W
                im = self.featuresReshape(im, N, C, H, W)
                self.mn = torch.mean(im, dim=1, keepdim=True)
                # Centering the data
                im = im - self.mn
                self.get_stats_params(im)
                # projection
                imProj = torch.matmul(self.u.t(), im)

                # conv + bn if exists + projection
                im2 = F.conv2d(input, self.weight, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)
                imProj2 = self.featuresReshape(im2, N, C, H, W)
                assert (torch.max(torch.abs(imProj - imProj2)) < 0.1)
            else:
                # conv + bn if exists + projection
                im = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
                N, C, H, W = im.shape  # N x C x H x W
                imProj = self.featuresReshape(im, N, C, H, W)

            mult = torch.zeros(1).to(imProj)
            add = torch.zeros(1).to(imProj)

            if self.collectStats:
                # collect b of laplacian distribution
                if self.stats == 'channel':
                    for i in range(0, self.channels):
                        self.lapB[i] += torch.sum(torch.abs(imProj[i, :]))
                        self.numElems[i] += imProj.shape[1]
                        self.gausStd[i] = torch.std(imProj[i, :])
                elif self.stats == 'first':
                    self.lapB += torch.sum(torch.abs(imProj[0, :]))
                    self.numElems += imProj.shape[1]
                    self.gausStd += torch.std(imProj[0, :])
                elif self.stats == 'all':
                    self.lapB += torch.sum(torch.abs(imProj))
                    self.numElems += imProj.numel()
                    self.gausStd += torch.std(imProj.view(-1))
                else:
                    raise ValueError("Wrong stats type")

                self.collectStats = False

            dynMax = torch.max(imProj)
            dynMin = torch.min(imProj)

            if self.actBitwidth < 30:
                imProj, mult, add = part_quant(imProj, max=dynMax, min=dynMin,
                                               bitwidth=self.actBitwidth)

            self.act_size = imProj.numel()
            self.bit_per_entry = shannon_entropy(imProj).item()
            self.bit_count = self.bit_per_entry * self.act_size

            if False:
                self.bit_countH = huffman_encode(imProj)
                self.bit_per_entryH = self.bit_countH / self.act_size

            # if self.actBitwidth < 30:
            #     for i in range(0, self.channels):
            #         imProj[i, :] = imProj[i, :] * mult[i] + add[i]

            if self.actBitwidth < 30:
                imProj = imProj * mult + add

            imProj = torch.matmul(self.u, imProj)

            # Bias Correction
            imProj = imProj - torch.mean(imProj, dim=1, keepdim=True)

            # self.mse = torch.sum((imProj - im) ** 2)
            # return original mean
            imProj = imProj + self.mn

            # return to general
            #   input = imProj.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
            #  input = imProj.contiguous().view(N, C, H, W)# N x C x H x W
            input = self.featuresReshapeBack(imProj, N, C, H, W)
        else:
            input = F.conv2d(input, self.weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)

        # input = self.relu(input)
        return input

    def updateClamp(self, data):
        self.lapB[0] = (self.lapB[0] / self.numElems[0])
        self.clampValL[0] = opt.minimize_scalar(
            lambda x: mse_laplace(x, b=self.lapB[0].item(), num_bits=self.actBitwidth)).x

        self.clampVal = self.clampValL[0]

        # origDist = torch.histc(data.cpu())
        # origDist = origDist / torch.sum(origDist)
        #
        # clamps = np.linspace(0.1, 2, 40) * self.clampValL[0].item()
        #
        # mses = []
        # divr = []
        # entr = []
        #
        # for c in range(0, len(clamps)):
        #
        #     imProj = torch.matmul(self.u.t(), data)
        #     clampMax = clamps[c]
        #     clampMin = -clampMax
        #
        #     if clampMax < torch.max(imProj) or clampMin > torch.min(imProj):
        #         imProj = torch.clamp(imProj, max=clampMax, min=clampMin)
        #
        #         dynMax = torch.max(imProj)
        #         dynMin = torch.min(imProj)
        #         if self.actBitwidth < 30:
        #             imProj, mult, add = part_quant(imProj, max=dynMax, min=dynMin,
        #                                            bitwidth=self.actBitwidth)
        #
        #         act_size = imProj.numel()
        #         bit_per_entry = shannon_entropy(imProj).item()
        #         bit_count = bit_per_entry * act_size
        #         #    print(bit_per_entry)
        #
        #         if self.actBitwidth < 30:
        #             imProj = imProj * mult + add
        #
        #         imProj = torch.matmul(self.u, imProj)
        #
        #         # Bias Correction
        #         imProj = imProj - torch.mean(imProj, dim=1, keepdim=True)
        #
        #         currDist = torch.histc(imProj.cpu(), max=torch.max(data), min=torch.min(data))
        #         currDist = currDist / torch.sum(currDist)
        #
        #         currDiv = origDist * (-1) * torch.log(origDist / currDist)
        #         currDiv[origDist == 0] = 0
        #         currDiv[currDist == 0] = 0
        #         divr.append(torch.sum(currDiv))
        #         mses.append(torch.mean((imProj - data) ** 2))
        #         entr.append(bit_per_entry)
        #
        # divr = torch.tensor(divr)
        # mses = torch.tensor(mses)
        # entr = torch.tensor(entr)
        #
        # # print('*****')
        # # print(divr)
        # # print(mses)
        # # print(entr)
        #
        # # idxs = (divr == torch.max(divr)).nonzero()
        # # idx = torch.argmin(mses[idxs])
        #
        # idxs = (mses == torch.min(mses)).nonzero()
        # idx = torch.argmax(divr[idxs])
        #
        # #  print(idxs[idx])
        #
        # self.clampVal = torch.tensor(clamps[idxs[idx]])
        #


# taken from https://github.com/submission2019/cnn-quantization/blob/master/optimal_alpha.ipynb
def mse_laplace(alpha, b, num_bits):
    #  return 2 * (b ** 2) * np.exp(-alpha / b) + ((alpha ** 2) / (3 * 2 ** (2 * num_bits)))
    exp_val = 1e300 if -alpha / b > 690 else np.exp(-alpha / b)  # prevent overflow
    # res = (b ** 2) * exp_val + ((alpha ** 2) / (24 * 2 ** (2 * num_bits))) #Fused relu
    res = 2 * (b ** 2) * exp_val + ((alpha ** 2) / (3 * 2 ** (2 * num_bits)))
    return res


def mse_gaussian(alpha, sigma, num_bits):
    clipping_err = (sigma ** 2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma * np.sqrt(2.0)))) - \
                   np.sqrt(2.0 / np.pi) * alpha * sigma * (np.e ** ((-1) * (0.5 * (alpha ** 2)) / sigma ** 2))
    quant_err = (alpha ** 2) / (3 * (2 ** (2 * num_bits)))
    return clipping_err + quant_err


def part_quant(x, max, min, bitwidth):
    if max != min:
        act_scale = (2 ** bitwidth - 1) / (max - min)
        q_x = Round.apply((x - min) * act_scale)
        return q_x, 1 / act_scale, min
    else:
        q_x = x
        return q_x, 1, 0


def part_quant_force_zero(x, max, min, bitwidth):
    if max != min:
        act_scale = (2 ** bitwidth - 1) / (max - min)
        initial_zero_point = torch.tensor(-min * act_scale)
        # make zero exactly represented
        zero_point = Round.apply(initial_zero_point)
        q_x = Round.apply(x * act_scale + zero_point)
        return q_x, 1 / act_scale, zero_point
    else:
        q_x = x
    return q_x, 1, 0


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
        round = x.round()
        return round.to(x.device)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input, None, None
