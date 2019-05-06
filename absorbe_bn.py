import torch
import torch.nn as nn


def absorb_bn(module, bn_module):
    if module.bias is None:
        zeros = torch.Tensor(module.out_channels).zero_().type(module.weight.data.type())
        module.bias = nn.Parameter(zeros)
    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
    module.weight.data.mul_(invstd.view(module.weight.data.size(0), 1, 1, 1).expand_as(module.weight.data))
    module.bias.data.add_(-bn_module.running_mean).mul_(invstd)

    if bn_module.affine:
        module.weight.data.mul_(bn_module.weight.data.view(module.weight.data.size(0), 1, 1, 1).expand_as(module.weight.data))
        module.bias.data.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    bn_module.register_buffer('running_mean', torch.zeros(module.out_channels).cuda())
    bn_module.register_buffer('running_var', torch.ones(module.out_channels).cuda())
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)
    bn_module.affine = False


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d) and m.groups == 1) or isinstance(m, nn.Linear)


def search_absorbe_bn(model):
    prev = None
    for m in model.children():
        if is_bn(m):
            if is_absorbing(prev):
                m.absorbed = True
                absorb_bn(prev, m)
            else:
                m.absorbed = False

        search_absorbe_bn(m)
        prev = m
