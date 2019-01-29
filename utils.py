import os
import numpy as np
import torch
from shutil import copyfile
import logging
from inspect import getfile, currentframe, isclass
from os import path, listdir, walk
from smtplib import SMTP
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from base64 import b64decode
from zipfile import ZipFile, ZIP_DEFLATED
from json import dump

from torch.autograd import Variable
from torch import save as saveModel
from torch import load as loadModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import tqdm
import torchvision
from os.path import join
import torchvision.datasets as datasets
import Models as models
import collections
import actquant

__DATASETS_DEFAULT_PATH = '/media/ssd/Datasets/'




def loadCheckpoint(dataset, model, bitwidth, filename='model_opt.pth.tar'):
    # init project base folder
    baseFolder = path.dirname(path.abspath(getfile(currentframe())))  # script directory
    # init checkpoint key
    checkpointKey = '[{}],[{}]'.format(bitwidth, model)
    # init checkpoint path
    checkpointPath = '{}/../pre_trained/{}/train_portion_1.0/{}/train/{}'.format(baseFolder, dataset, checkpointKey, filename)
    # load checkpoint
    checkpoint = None
    if path.exists(checkpointPath):
        checkpoint = loadModel(checkpointPath, map_location=lambda storage, loc: storage.cuda())

    return checkpoint, checkpointKey


def logBaselineModel(args, logger):
    # get baseline bitwidth
    bitwidth = args.baselineBits[0]
    # load baseline checkpoint
    checkpoint, checkpointKey = loadCheckpoint(args.dataset, args.model, bitwidth)
    # init logger rows
    loggerRows = []
    # init best_prec1 values
    best_prec1_str = 'Not found'

    if checkpoint is not None:
        keysFromUniform = ['epochs', 'learning_rate']
        # extract keys from uniform checkpoint
        for key in keysFromUniform:
            if key in checkpoint:
                value = checkpoint.get(key)
                if args.copyBaselineKeys:
                    setattr(args, key, value)
                    if logger:
                        loggerRows.append(['Loaded key [{}]'.format(key), value])
                elif logger:
                    loggerRows.append(['{}'.format(key), '{}'.format(value)])
        # extract best_prec1 from uniform checkpoint
        best_prec1 = checkpoint.get('best_prec1')
        if best_prec1:
            best_prec1_str = '{:.3f}'.format(best_prec1)

    # print result
    if logger:
        loggerRows.append(['Model', '{}'.format(checkpointKey)])
        loggerRows.append(['Validation accuracy', '{}'.format(best_prec1_str)])
        logger.addInfoTable('Baseline model', loggerRows)


def check_if_need_to_collect_statistics(model):
    for layer in model.modules():
    # for layer in model.module.layers_list():
        if isinstance(layer, actquant.ActQuantBuffers):
            if hasattr(layer, 'running_std') and float(layer.running_std) != 0:
                return False

    return True


# collect possible models names
def loadModelNames():
    return [name for (name, obj) in models.__dict__.items() if isclass(obj)]


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def saveArgsToJSON(args):
    # save args to JSON
    args.jsonPath = '{}/args.txt'.format(args.save)
    with open(args.jsonPath, 'w') as f:
        dump(vars(args), f, indent=4, sort_keys=True)



def zipFolder(p, pathFunc, zipf):
    for base, dirs, files in walk(p):
        if base.endswith('__pycache__'):
            continue

        for file in files:
            if file.endswith('.tar'):
                continue

            fn = path.join(base, file)
            zipf.write(fn, pathFunc(fn))


def zipFiles(saveFolder, zipFname, attachPaths):
    zipPath = '{}/{}'.format(saveFolder, zipFname)
    zipf = ZipFile(zipPath, 'w', ZIP_DEFLATED)
    for p in attachPaths:
        if path.exists(p):
            if path.isdir(p):
                zipFolder(p, zipf)
            else:
                zipf.write(p)
    zipf.close()

    return zipPath





def create_exp_dir(resultFolderPath):
    # create folders
    if not os.path.exists(resultFolderPath):
        os.makedirs(resultFolderPath)

    codeFilename = 'code.zip'
    zipPath = '{}/{}'.format(resultFolderPath, codeFilename)
    zipf = ZipFile(zipPath, 'w', ZIP_DEFLATED)

    # init project base folder
    baseFolder = path.dirname(path.abspath(getfile(currentframe())))  # script directory
    baseFolder += '/..'
    # init path function
    pathFunc = lambda fn: path.relpath(fn, baseFolder)
    # init folders we want to zip
    foldersToZip = ['Models']
    # save folders files
    for folder in foldersToZip:
        zipFolder('{}/{}'.format(baseFolder, folder), pathFunc, zipf)

    # save cnn folder files
    foldersToZip = ['FeaturesCompress']
    for folder in foldersToZip:
        folderName = '{}/{}'.format(baseFolder, folder)
        for file in listdir(folderName):
            filePath = '{}/{}'.format(folderName, file)
            if path.isfile(filePath):
                zipf.write(filePath, pathFunc(filePath))

    # close zip file
    zipf.close()

    return zipPath, codeFilename


checkpointFileType = 'pth.tar'
stateFilenameDefault = 'model'
stateCheckpointPattern = '{}/{}_checkpoint.' + checkpointFileType
stateOptModelPattern = '{}/{}_opt.' + checkpointFileType


def save_state(state, is_best, path, filename):
    default_filename = stateCheckpointPattern.format(path, filename)
    saveModel(state, default_filename)

    is_best_filename = None
    if is_best:
        is_best_filename = stateOptModelPattern.format(path, filename)
        copyfile(default_filename, is_best_filename)

    return default_filename, is_best_filename


def save_checkpoint(path, model, args, epoch, best_prec1, is_best=False, filename=None):
    print('*** save_checkpoint ***')
    # we want to save full-precision weights, we will quantize them after loading them
    model.removeQuantizationFromStagedLayers()
    # set state dictionary
    state = dict(nextEpoch=epoch + 1, state_dict=model.state_dict(), epochs=args.epochs, alphas=model.save_alphas_state(), updated_statistics=False,
                 nLayersQuantCompleted=model.nLayersQuantCompleted, best_prec1=best_prec1, learning_rate=args.learning_rate)
    # set state filename
    filename = filename or stateFilenameDefault
    # save state to file
    filePaths = save_state(state, is_best, path=path, filename=filename)

    # restore quantization in staged layers
    model.restoreQuantizationForStagedLayers()
    print('*** END save_checkpoint ***')

    return state, filePaths


def setup_logging(log_file, logger_name, propagate=False):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    # logging to stdout
    logger.propagate = propagate

    return logger


def initLogger(folderName, propagate=False):
    filePath = '{}/log.txt'.format(folderName)
    logger = setup_logging(filePath, 'darts', propagate)

    logger.info('Experiment dir: [{}]'.format(folderName))

    return logger


def initTrainLogger(logger_file_name, folder_path, propagate=False):
    # folder_path = '{}/train'.format(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    log_file_path = '{}/{}.txt'.format(folder_path, logger_file_name)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    logger = setup_logging(log_file_path, logger_file_name, propagate)

    return logger


def loadDatasets():
    return dict(cifar10=10, cifar100=100, imagenet=1000)





def get_dataset(name, train, transform, target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    root = datasets_path  # '/mnt/ssd/ImageNet/ILSVRC/Data/CLS-LOC' #os.path.join(datasets_path, name)

    if name == 'cifar10':
        cifar_ = datasets.CIFAR10(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        return cifar_

    elif name == 'cifar100':
        cifar_ = datasets.CIFAR100(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        return cifar_

    elif name == 'imagenet':
        if train:
            root = join(root, 'train')
        else:
            root = join(root, 'val')

        return datasets.ImageFolder(root=root, transform=transform, target_transform=target_transform)



def load_data(args,logger):
    # init transforms
    logger.info('==> Preparing data..')
    #TODO - do different transform to each dataset
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

    transform = {'train': transform_train , 'test': transform_test}

    train_data = get_dataset(args.dataset, train=True, transform=transform['train'], datasets_path=args.data)
    test_data = get_dataset(args.dataset, train=False, transform=transform['test'], datasets_path=args.data)


    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=2)


    return trainloader, testloader


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


# msg - email message, MIMEMultipart() object
def attachFiletoEmail(msg, fileFullPath):
    with open(fileFullPath, 'rb') as z:
        # attach file
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(z.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % path.basename(fileFullPath))
        msg.attach(part)


def sendEmail(toAddr, subject, content, attachments=None):
    # init email addresses
    fromAddr = "brianch@campus.technion.ac.il"
    # init connection
    server = SMTP('smtp.office365.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    passwd = b'WXo4Nzk1NzE='
    server.login(fromAddr, b64decode(passwd).decode('utf-8'))
    # init message
    msg = MIMEMultipart()
    msg['From'] = fromAddr
    msg['Subject'] = subject
    msg.attach(MIMEText(content, 'plain'))

    if attachments:
        for att in attachments:
            if path.exists(att):
                if path.isdir(att):
                    for filename in listdir(att):
                        attachFiletoEmail(msg, '{}/{}'.format(att, filename))
                else:
                    attachFiletoEmail(msg, att)

    # send message
    for dst in toAddr:
        msg['To'] = dst
        text = msg.as_string()
        try:
            server.sendmail(fromAddr, dst, text)
        except Exception as e:
            print('Sending email failed, error:[{}]'.format(e))

    server.close()


def sendDataEmail(model, args, logger, content):
    # init files to send
    attachments = [model.alphasCsvFileName, model.stats.saveFolder, args.jsonPath, model._criterion.bopsLossImgPath, logger.fullPath]
    # init subject
    subject = 'Results [{}] - Model:[{}] Bitwidth:{} dataset:[{}] lambda:[{}]' \
        .format(args.folderName, args.model, args.bitwidth, args.dataset, args.lmbda)
    # send email
    sendEmail(args.recipients, subject, content, attachments)

    #
    # def logParameters(logger, args, model):
    #     if not logger:
    #         return
    #
    #     # calc number of permutations
    #     permutationStr = model.nPerms
    #     for p in [12, 9, 6, 3]:
    #         v = model.nPerms / (10 ** p)
    #         if v > 1:
    #             permutationStr = '{:.3f} * 10<sup>{}</sup>'.format(v, p)
    #             break
    #     # log other parameters
    #     logger.addInfoTable('Parameters', HtmlLogger.dictToRows(
    #         {
    #             'Parameters size': '{:.3f} MB'.format(count_parameters_in_MB(model)),
    #             'Learnable params': len(model.learnable_params),
    #             'Ops per layer': [layer.numOfOps() for layer in model.layersList],
    #             'Permutations': permutationStr
    #         }, nElementPerRow=2))
    #     # log baseline model
    #     logBaselineModel(args, logger)
    #     # log args
    #     logger.addInfoTable('args', HtmlLogger.dictToRows(vars(args), nElementPerRow=3))
    #     # print args
    #     print(args)
