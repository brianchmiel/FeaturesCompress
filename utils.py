import logging
import random
from json import dump
from os.path import join

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import Models

__DATASETS_DEFAULT_PATH = './data/'


def checkModelDataset(args):
    if args.dataset == 'imagenet':
        assert (args.model != 'resnet20' and args.model != 'resnet56')
    else:
        assert (args.model != 'resnet18' and args.model != 'resnet50')

    # collect possible models names


def loadModelNames():
    return [name for (name, obj) in Models.__dict__.items() if hasattr(obj, '__call__')]


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def saveArgsToJSON(args):
    # save args to JSON
    args.jsonPath = '{}/args.txt'.format(args.save)
    with open(args.jsonPath, 'w') as f:
        dump(vars(args), f, indent=4, sort_keys=True)


def loadDatasets():
    return dict(cifar10=10, cifar100=100, imagenet=1000)


def get_dataset(name, train, transform, target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    root = datasets_path  # '/mnt/ssd/ImageNet/ILSVRC/Data/CLS-LOC' #os.path.join(datasets_path, name)
    if name == 'cifar10':
        cifar_ = datasets.CIFAR10(root=root, train=train, transform=transform, target_transform=target_transform,
                                  download=download)
        return cifar_

    elif name == 'cifar100':
        cifar_ = datasets.CIFAR100(root=root, train=train, transform=transform, target_transform=target_transform,
                                   download=download)
        return cifar_

    elif name == 'imagenet':
        if train:
            root = join(root, 'train')
        else:
            root = join(root, 'val')
        return datasets.ImageFolder(root=root, transform=transform, target_transform=target_transform)


def get_transform(args):
    if args.dataset == 'imagenet':
        resize = 256 if args.model != 'inception_v3' else 299
        crop_size = 224 if args.model != 'inception_v3' else 299
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(crop_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])
    else:  # cifar
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return transform_train, transform_test

def measureCorrBetweenChannels(tensor):
    N, C, H, W = tensor.shape  # N x C x H x W
    tensor = tensor.detach().transpose(0, 1).contiguous()
    tensor = tensor.contiguous().view(tensor.shape[0], -1)
    mn = torch.mean(tensor, dim=1, keepdim=True)
    # Centering the data
    tensor = tensor - mn
    sm = torch.zeros(1)
    elems = torch.zeros(1)
    corr = torch.zeros((C,C))
    cov = torch.matmul(tensor, tensor.t()) / tensor.shape[1]
    # svd
    u, s, _ = torch.svd(cov)
    tensor = torch.matmul(u.t(),tensor)
    for i in range(0,C):
        for j in range(0,i):
            data1 = tensor[i,:]
            data2 = tensor[j,:]
            cov = (data1.dot(data2)) / data1.shape[0]
            std1 = torch.std(data1)
            std2 = torch.std(data2)
            corr[i,j] = cov / (std1 * std2)
            sm += torch.abs(corr[i,j])
            elems += 1
    print(((sm + C) / (elems)).item())
    return corr
def load_data(args, logger):
    # init transforms
    logger.info('==> Preparing data..')
    transform_train, transform_test = get_transform(args)

    transform = {'train': transform_train, 'test': transform_test}

    train_data = get_dataset(args.dataset, train=True, transform=transform['train'], datasets_path=args.data)
    test_data = get_dataset(args.dataset, train=False, transform=transform['test'], datasets_path=args.data)

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # sample = SubsetRandomSampler(np.linspace(0,40000,40000+1,dtype = np.int)[:-1])
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=True, num_workers=2)

    # statsdata = get_dataset(args.dataset, train=False, transform=transform['test'], datasets_path=args.data)

    statsBatchSize = args.batch * 2
    data_len = 50000 if args.dataset == 'imagenet' else 10000
    rndIdx = random.randint(0, data_len - statsBatchSize)
    sample = SubsetRandomSampler(np.linspace(rndIdx, rndIdx + statsBatchSize, statsBatchSize + 1, dtype=np.int)[:-1])

    statloader = torch.utils.data.DataLoader(test_data, batch_size=statsBatchSize, shuffle=False, num_workers=2,
                                             sampler=sample) # TODO

    return trainloader, testloader, statloader


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
