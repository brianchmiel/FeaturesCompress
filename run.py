import tqdm
import torch
import os



class Run:
    def __init__(self, model, logging, optimizer, criterion,best_acc):
        self.model = model
        self.logging = logging
        self.optimizer = optimizer
        self.criterion = criterion
        self.best_acc = best_acc

    def runTrain(self, trainLoader):
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainLoader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            out = self.model(inputs)
            loss = self.criterion(out, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.logging.info('step: {} / {} : Loss: {:.3f} '
                         '| Acc: {:.3f}% ({}/{})'.format(batch_idx, len(trainLoader), train_loss / (batch_idx + 1),
                                                         100. * correct / total, correct, total))

    def runTest(self, args, testLoader, epoch):
        self.model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(testLoader)):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = self.model(inputs)
                loss = self.criterion(out, targets)

                test_loss += loss.item()
                _, predicted = out.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                self.logging.info('step: {} / {} : Loss: {:.3f} | Acc: {:.3f}% ({}/{}) '
                             .format(batch_idx, len(testLoader), test_loss / (batch_idx + 1),
                                                    100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            self.logging.info('Saving..')
            state = {
                'net': self.model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('results'):
                os.mkdir('results')
            torch.save(state, args.save + '/ckpt.t7')
            self.best_acc = acc
