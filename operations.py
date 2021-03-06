import contextlib
import math
import os
import tempfile

import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn
from tqdm import trange, tqdm

from ar_encoder.arithmeticcoding import SimpleFrequencyTable, ArithmeticEncoder, BitOutputStream
from entropy import shannon_entropy
from huffman import huffman_encode
from utils import write_int

eps = 1e-6
reps = 1e-2  # empirical value


def small(x):
    return torch.abs(x) < reps * torch.max(torch.abs(x))


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


def get_projection_matrix(im, projType):
    if projType == 'pca':
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        # svd
        u, s, _ = torch.svd(cov)
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
    def __init__(self, args, ch):
        super(ReLuPCA, self).__init__()

        self.size_from_ar_encoder = False
        self.size_from_huff_encoder = True

        self.actBitwidth = args.actBitwidth
        self.projType = args.projType
        self.per_channel = args.perCh
        if self.projType == 'eye':
            self.stats = 'all' if not self.per_channel else 'channel'
        else:
            self.stats = 'first' if not self.per_channel else 'channel'

        self.collectStats = True
        self.bit_count = None
        self.relu = nn.ReLU(inplace=True)
        self.microBlockSz = args.MicroBlockSz
        self.channelsDiv = args.channelsDiv

        dataBasicSize = 7 if args.dataset == 'imagenet' else 2
        if self.microBlockSz > 1:
            assert (self.microBlockSz % dataBasicSize == 0)
        #   assert (self.channelsDiv % 2 == 0)

    def featuresReshape(self, input, N, C, H, W):
        # check input
        if (self.microBlockSz > H):
            self.microBlockSz = H
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

        N, C, H, W = input.shape  # N x C x H x W
        input = self.relu(input)

        #  im = input.detach().transpose(0, 1).contiguous()
        #  im = im.view(im.shape[0], -1)
        im = self.featuresReshape(input, N, C, H, W)

        self.channels = im.shape[0]
        if not hasattr(self, 'clampVal'):
            self.register_buffer('clampVal', torch.zeros(self.channels))
        if not hasattr(self, 'lapB'):
            self.register_buffer('lapB', torch.zeros(self.channels))
        if not hasattr(self, 'numElems'):
            self.register_buffer('numElems', torch.zeros(self.channels))

        mn = torch.mean(im, dim=1, keepdim=True)
        # Centering the data
        im = im - mn

        # Calculate projection matrix if needed
        if self.collectStats:
            self.u, self.s = get_projection_matrix(im, self.projType)

        # projection
        imProj = torch.matmul(self.u.t(), im)

        mult = torch.zeros(imProj.size(0)).to(imProj)
        add = torch.zeros(imProj.size(0)).to(imProj)
        if self.collectStats:
            # collect b of laplacian distribution
            if self.stats == 'channel':
                for i in range(0, self.channels):
                    self.lapB[i] += torch.sum(torch.abs(imProj[i, :]))
                    self.numElems[i] += imProj.shape[1]
            elif self.stats == 'first':
                self.lapB += torch.sum(torch.abs(imProj[0, :]))
                self.numElems += imProj.shape[1]
            elif self.stats == 'all':
                self.lapB += torch.sum(torch.abs(imProj))
                self.numElems += imProj.numel()
            else:
                raise ValueError("Wrong stats type")
            self.updateClamp()

        # quantize and send to memory
        for i in range(0, self.channels):
            clampMax = self.clampVal[i].item()
            clampMin = -self.clampVal[i].item()
            imProj[i, :] = torch.clamp(imProj[i, :], max=clampMax, min=clampMin)

        if self.stats == 'first' or self.stats == 'all':
            dynMax = torch.max(imProj)
            dynMin = torch.min(imProj)
        for i in range(0, self.channels):
            if self.stats == 'channel':
                dynMax = torch.max(imProj[i, :])
                dynMin = torch.min(imProj[i, :])

            if self.actBitwidth < 30:
                imProj[i, :], mult[i], add[i] = part_quant(imProj[i, :], max=dynMax, min=dynMin,
                                                           bitwidth=self.actBitwidth)

        self.act_size = imProj.numel()
        if self.size_from_ar_encoder:
            int_img = torch.round(imProj).long().flatten()
            counts = torch.bincount(int_img)
            freqs = list(counts.cpu().numpy()) + [1]
            afreqs = SimpleFrequencyTable(freqs)
            int_img = int_img.cpu().numpy()
            # todo don't write to disk
            with tempfile.TemporaryFile() as fp:
                with contextlib.closing(BitOutputStream(fp)) as bitout:
                    for i in range(len(freqs)):
                        write_int(bitout, 32, afreqs.get(i))

                    enc = ArithmeticEncoder(32, bitout)
                    for symbol in int_img:
                        enc.write(afreqs, symbol)
                    enc.write(afreqs, len(freqs) - 1)  # EOF
                    enc.finish()  # Flush remaining code bits

                    fp.seek(0, os.SEEK_END)
                    size = fp.tell()
            self.actual_bit_count = self.actBitwidth * self.act_size / 8
            self.bit_per_entry = size * 8
            self.bit_count = self.bit_per_entry * self.act_size
            print("{} bytes compressing {} bytes. {} compression".format(size, self.actual_bit_count,
                                                                         size / self.actual_bit_count))

        elif self.size_from_huff_encoder:
            int_img = torch.round(imProj).long().flatten()
            counts = torch.bincount(int_img)
            freqs = list(counts.cpu().numpy())
            codes = huffman_encode(freqs)
            self.bit_count = np.sum([len(codes[i]) * freqs[i] for i in range(len(freqs))])
            self.bit_per_entry = self.bit_count / self.act_size

        else:
            self.bit_per_entry = shannon_entropy(imProj).item()
            self.bit_count = self.bit_per_entry * self.act_size

        if self.actBitwidth < 30:
            for i in range(0, self.channels):
                imProj[i, :] = imProj[i, :] * mult[i] + add[i]
        imProj = torch.matmul(self.u, imProj)

        # Bias Correction
        imProj = imProj - torch.mean(imProj, dim=1, keepdim=True)

        # return original mean
        imProj = imProj + mn

        # return to general
        # input = imProj.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
        # input = imProj.view(N, C, H, W).contiguous()  # N x C x H x W
        input = self.featuresReshapeBack(imProj, N, C, H, W)

        self.collectStats = False
        return input

    def updateClamp(self):
        for i in range(0, self.channels):
            self.lapB[i] = (self.lapB[i] / self.numElems[i])
            if self.lapB[i] > 0:
                self.clampVal[i] = opt.minimize_scalar(
                    lambda x: mse_laplace(x, b=self.lapB[i].item(), num_bits=self.actBitwidth)).x
            else:
                self.clampVal[i] = 0


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


def part_quant(x, max, min, bitwidth):
    if max != min:
        act_scale = (2 ** bitwidth - 2) / (max - min)
        q_x = Round.apply((x - min) * act_scale)
        return q_x, 1 / act_scale, min
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
