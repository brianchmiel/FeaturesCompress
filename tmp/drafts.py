# class partialConv2d(Mconv._ConvNd):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         super(partialConv2d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             False, _pair(0), groups, bias)
#         self.weightShape = self.weight.shape
#     def forward(self, input):
#         for i in range(0,input.shape[1]):
#             if i == 0:
#                 out = F.conv2d(input[:,i,:,:].unsqueeze_(1), self.weight[:,i,:,:].view(self.weightShape[0],1,self.weightShape[2],self.weightShape[3]), self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#             else:
#                 out += F.conv2d(input[:,i,:,:].unsqueeze_(1), self.weight[:,i,:,:].view(self.weightShape[0],1,self.weightShape[2],self.weightShape[3]), self.bias, self.stride,
#                             self.padding, self.dilation, self.groups)
#
#
#
#         return out

class partialConv2d(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        # kernel_sizeP = _pair(kernel_size)
        # strideP = _pair(stride)
        # paddingP = _pair(padding)
        # dilationP = _pair(dilation)
        super(partialConv2d, self).__init__()
        self.conv = []
        self.compress = Compression(args)
        for i in range(0, in_channels):
            self.conv.append(nn.Conv2d(1, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias).cuda())
        self.in_channels = in_channels

    def forward(self, input):
        for i in range(0, input.shape[1]):
            if i == 0:
                out = self.conv[i](input[:, i, :, :].unsqueeze_(1))
            else:
                out += self.conv[i](input[:, i, :, :].unsqueeze_(1))
            out = self.compress(out)
        return out

    def loadPreTrained(self, w, b=None):
        for i in range(0, self.in_channels):
            self.conv[i].weight.data = w[:, i, :, :].unsqueeze_(1)
            if b is not None:
                self.conv[i].bias.data = b[:, i].unsqueeze_(1)


class Compression(nn.Module):
    def __init__(self, args):
        super(Compression, self).__init__()
        self.MacroMicroRatio = args.MacroBlockSz / args.MicroBlockSz
        self.MacroSz = args.MacroBlockSz
        self.MicroSz = args.MicroBlockSz
        #  self.macroHad = torch.from_numpy(hadamard(args.MacroBlockSz))
        self.microHad = torch.from_numpy(hadamard(args.MicroBlockSz)).float().cuda()
        if args.FixedQuant:
            # TODO - add fixed quantization
            self.quantAC = torch.ones((1, 15))
            self.quantDC = torch.ones((1, 16))
        else:
            # TODO - add option of not fixed quantization
            pass

    def forward(self, input):
        compressed = self.encoder(input)
        # TODO - Add Loss
        return self.decoder(compressed)

    def encoder(self, input):
        rows = input.shape[2] / self.MacroSz
        columns = input.shape[3] / self.MacroSz
        # Divide input to Macro Block
        for i in range(int(rows)):
            for j in range(int(columns)):
                dcMacroCoeff = []
                macroBlockRow = i * self.MacroSz
                macroBlockColumn = j * self.MacroSz
                for k in range(int(self.MacroMicroRatio)):
                    microBlockRowShift = int(k / self.MicroSz)
                    microBlockColumnShift = (k % self.MicroSz) * self.MicroSz
                    # each macro block divide to micro blocks
                    hadMatrix = self.hadamardT(
                        input[:, :,
                        macroBlockRow + microBlockRowShift: macroBlockRow + microBlockRowShift + self.MicroSz,
                        macroBlockColumn + microBlockColumnShift:macroBlockColumn + microBlockColumnShift + self.MicroSz])
                    # Union all DC Coeff
                    dcMacroCoeff.append(hadMatrix[0][0])
                    # Flatten AC Coeff
                    acMicroCoeff = hadMatrix.view(-1)
                    # Remove DC from AC coeff
                    acMicroCoeff = acMicroCoeff[1:]
                    # Quantization of AC
                    acMicroCoeff = torch.round(acMicroCoeff / self.quantAC)
                    # TODO - AC Lossless compression : arithmetic or entropy encoder

                # DC Compress again

                # DC Quantization
                dcMacroCoeff = torch.round(dcMacroCoeff / self.quantDC)

                # TODO - DC Lossless Compression

    def decoder(self, input):
        pass
        # TODO - finish decoder

    def hadamardT(self, input):
        return (1 / self.MicroSz) * torch.matmul(torch.matmul(self.microHad, input), self.microHad.t())

    def hadamardInverseT(self, input):
        return (1 / self.MicroSz) * torch.matmul(torch.matmul(self.microHad.t(), input), self.microHad)
