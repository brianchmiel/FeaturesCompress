'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from sys import exit, argv
from os import getpid, environ
import argparse
import logging
import os
from os.path import dirname, abspath
from inspect import getfile, currentframe, isclass
import time
from json import dump
from traceback import format_exc
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed
import numpy as np
import torch.backends.cudnn as cudnn
from datetime import datetime
from socket import gethostname
import torch
import torch.optim as optim
import tqdm
from torch.nn import CrossEntropyLoss
from utils import loadModelNames, loadDatasets, saveArgsToJSON, TqdmLoggingHandler, sendEmail, load_data, models, check_if_need_to_collect_statistics
from run import Run
from compress_loss import CompressLoss
def parseArgs():

    modelNames = loadModelNames()
    datasets = loadDatasets()


    parser = argparse.ArgumentParser(description='FeaturesCompress')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--data', type=str, default ='./data/',help='location of the data corpus' )
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')

    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=datasets.keys(), help='dataset name')

    parser.add_argument('--preTrained', type=str, default=None, help='pre-trained model to copy weights from')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--actBitwidth', default=32, type=int, metavar='N',
                        help='Quantization activation bitwidth (default: 5)')
    parser.add_argument('--model', '-a', metavar='MODEL', default='tiny_resnet', choices=modelNames,
                        help='model architecture: ' + ' | '.join(modelNames) + ' (default: tinyresnet)')
    parser.add_argument('--epochs', type=int, default=10,help='num of training epochs ')
    parser.add_argument('--compressRatio', type=int, default=None, help='Compress Ratio')
    parser.add_argument('--FixedQuant', type= bool, default = True, help='Use fixed quantization?')
    parser.add_argument('--MacroBlockSz', type=int, default=16, help='MacroBlockSz')
    parser.add_argument('--MicroBlockSz', type=int, default=4, help='MicroBlockSz')
    parser.add_argument('--EigenVar', type=float, default=0.98, help='EigenVar')
    parser.add_argument('--lmbda', type=float, default=100, help='Lambda value for CompressLoss')
    parser.add_argument('--layerCompress', type=int, default=3, help='Which layer we want to add compression')
    args = parser.parse_args()


    # update GPUs list
    if type(args.gpu) is str:
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    # set number of model output classes
    args.nClasses = datasets[args.dataset]

    # create folder
    baseFolder = dirname(abspath(getfile(currentframe())))
    args.time = time.strftime("%Y%m%d-%H%M%S")
    args.folderName = '{}_{}_{}'.format(args.model, args.dataset, args.time)
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

    #CUDA
    args.seed = datetime.now().microsecond
    np.random.seed(args.seed)
    set_device(args.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)

    #Logger

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename=os.path.join(args.save, 'log.txt'), level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    th = TqdmLoggingHandler()
    th.setFormatter(logging.Formatter(log_format))
    log = logging.getLogger()
    log.addHandler(th)


    # Data
    trainLoader, testLoader = load_data(args, logging)

    # Model
    logging.info('==> Building model..')
    modelClass = models.__dict__[args.model]
    model = modelClass(args)
    model = model.cuda()

    #Parameters
    start_epoch = 0
    #preTrained
    if args.preTrained is not None:
        # Load checkpoint.
        logging.info('==> Resuming from checkpoint..')
        preTrainedDir = 'preTrained/' + args.preTrained
        assert os.path.isdir(preTrainedDir), 'Error: no checkpoint directory found!'
        model.loadPreTrained(preTrainedDir)


    #Optimization and criterion
    #TODO - add option to choose criterion and optimizer
    criterion = CompressLoss(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


    run =  Run(model, logging, optimizer, criterion)
    try:
        # log command line
        logging.info('CommandLine: {} PID: {} Hostname: {} CUDA_VISIBLE_DEVICES {}'.format(argv, getpid(), gethostname(), environ.get('CUDA_VISIBLE_DEVICES')))
        for epoch in tqdm.trange(start_epoch, start_epoch + args.epochs):
            logging.info('\nEpoch: {}'.format(epoch))
            # Train
            run.runTrain(trainLoader)
            # Test
            run.runTest(args, testLoader, epoch)
        logging.info('Done !')
    except Exception as e:
        # create message content
        # messageContent = '[{}] stopped due to error [{}] \n traceback:[{}]'. \
        #     format(args.folderName, str(e), format_exc())
        # # send e-mail with error details
        # subject = '[{}] stopped'.format(args.folderName)
        # sendEmail(['brianch@campus.technion.ac.il'], subject, messageContent)
        #
        # # forward exception
        raise e









