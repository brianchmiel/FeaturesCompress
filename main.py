from __future__ import print_function

import argparse
import logging
import os
import time
from datetime import datetime
from inspect import getfile, currentframe
from os import getpid, environ
from os.path import dirname, abspath
from socket import gethostname
from sys import exit, argv

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import tqdm
from torch import manual_seed as torch_manual_seed
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch.nn import CrossEntropyLoss

import Models
from run import Run
from utils import loadModelNames, loadDatasets, saveArgsToJSON, TqdmLoggingHandler, load_data, checkModelDataset


def parseArgs():
    modelNames = loadModelNames()
    datasets = loadDatasets()

    parser = argparse.ArgumentParser(description='FeaturesCompress')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--data', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')

    parser.add_argument('--batch', default=250, type=int, help='batch size')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=datasets.keys(), help='dataset name')

    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--actBitwidth', default=32, type=int, metavar='N',
                        help='Quantization activation bitwidth (default: 32)')
    parser.add_argument('--model', '-a', metavar='MODEL', choices=modelNames,
                        help='model architecture: ' + ' | '.join(modelNames))
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs ')
    parser.add_argument('--MicroBlockSz', type=int, default=1, help='MicroBlockSz')
    parser.add_argument('--EigenVar', type=float, default=1.0, help='EigenVar - should be between 0 to 1')
    parser.add_argument('--lmbda', type=float, default=0, help='Lambda value for CompressLoss')
    parser.add_argument('--projType', type=str, default='eye', choices=['eye', 'pca', 'optim'],
                        help='which projection we do: [eye, pca]')
    parser.add_argument('--clipType', type=str, default='laplace', choices=['laplace', 'gaussian'],
                        help='which clipping we do: [laplace, gaussian]')
    parser.add_argument('--project', action='store_true', help='if use projection - run only inference')
    parser.add_argument('--preTrained', action='store_true', help='pre-trained model to copy weights from')
    parser.add_argument('--perCh', action='store_true', help='per channel quantization')
    args = parser.parse_args()

    # check that model-dataset are good pair
    checkModelDataset(args)

    # update GPUs list
    if type(args.gpu) is not 'None':
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    # set number of model output classes
    args.nClasses = datasets[args.dataset]

    # create folder
    baseFolder = dirname(abspath(getfile(currentframe())))
    args.time = time.strftime("%Y%m%d-%H%M%S")
    args.folderName = '{}_{}_{}_{}_{}_{}'.format(args.model, args.projType, args.actBitwidth, args.EigenVar,
                                                 args.dataset, args.time)
    args.save = '{}/results/{}'.format(baseFolder, args.folderName)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # save args to JSON
    saveArgsToJSON(args)

    return args


if __name__ == '__main__':

    args = parseArgs()

    if not is_available():
        print('no gpu device available')
        exit(1)

    # CUDA
    args.seed = datetime.now().microsecond
    np.random.seed(args.seed)
    set_device(args.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)

    # Logger

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename=os.path.join(args.save, 'log.txt'), level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    th = TqdmLoggingHandler()
    th.setFormatter(logging.Formatter(log_format))
    log = logging.getLogger()
    log.addHandler(th)

    # Data
    trainLoader, testLoader, statloader = load_data(args, logging)

    # Model
    logging.info('==> Building model..')
    modelClass = Models.__dict__[args.model]
    model = modelClass(args)
    model = model.cuda()

    # Parameters
    start_epoch = 0
    # preTrained
    if args.preTrained:
        # Load checkpoint.
        logging.info('==> Resuming from checkpoint..')
        model.loadPreTrained()

    # Optimization and criterion
    # TODO - add option to choose criterion and optimizer
    criterion = CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    run = Run(model, logging, optimizer, criterion, args.lr)
    # log command line
    logging.info('CommandLine: {} PID: {} '
                 'Hostname: {} CUDA_VISIBLE_DEVICES {}'.format(argv, getpid(), gethostname(),
                                                               environ.get('CUDA_VISIBLE_DEVICES')))

    # collect statistics
    if args.project:
        logging.info('Starting collect statistics')
        model.enableStatisticPhase()
        run.collectStats(args, statloader, 0)
        model.disableStatisticPhase()
        logging.info('Finish collect statistics')
        logging.info('Run Projection on inference')
        run.runTest(args, testLoader, 0)
    else:
        for epoch in tqdm.trange(start_epoch, start_epoch + args.epochs):
            logging.info('\nEpoch: {}'.format(epoch))
            # Train
            #    run.runTrain(trainLoader, epoch)
            # Test
            run.runTest(args, testLoader, epoch)
    logging.info('Done !')
