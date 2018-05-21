#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.backcompat as backcompat
from torch.autograd import Variable

import targetprop as tp
from activations import Sign11, qReLU, ThresholdReLU
from models.cnn import CNN
from models.cnn_lstm import CNN_LSTM
from models.lstm import LSTM
from models.lstm_cnn import LSTM_CNN
from models.textcnn import TextCNN
from utils.datasethelper import create_datasets
from utils.metric import Metric
from utils.timercontext import timer_context

seed = 42
best_model_name = 'best_model.pth.tar'


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds', type=str, default='sentiment140', choices=('sentiment140', 'tsad'),
                        help='dataset on which to train')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='if specified, do not create a validation set and use it to choose the best model, '
                             'instead just use the model from the final iteration')

    parser.add_argument('--test-model', type=str,
                        help='specify the filename of a pre-trained model,  which will be loaded '
                             'and evaluated on the test set of the specified dataset')
    parser.add_argument('--arch', type=str, choices=('cnn', 'lstm', 'cnn-lstm', 'lstm-cnn', 'textcnn'),
                        help='model architecture to use')
    parser.add_argument('--nonlin', type=str, choices=('sign11', 'qrelu', 'relu', 'threshrelu'),
                        help='non-linearity to use in the specified architecture')
    parser.add_argument('--loss', type=str, default='crossent', choices=('crossent',),
                        help='the loss function to use for training')

    parser.add_argument('--tp-rule', type=str, default='SoftHinge', choices=[e.name for e in tp.TPRule],
                        help='the TargetProp rule to use')

    parser.add_argument('--batch', type=int, default=64,
                        help='batch size to use for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'rmsprop', 'adam'),
                        help='optimizer to use to train')
    parser.add_argument('--lr', type=float,
                        help='starting learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum amount for SGD')
    parser.add_argument('--wtdecay', type=float, default=0,
                        help='weight decay (L2 regularization) amount')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='factor by which to multiply the learning rate at each value in <lr-decay-epochs>')
    parser.add_argument('--lr-decay-epochs', type=int, nargs='+', default=None,
                        help='list of epochs at which to multiply the learning rate by <lr-decay>')

    parser.add_argument('--gpus', type=int, default=[0], nargs='+',
                        help='which GPU device ID(s) to train on')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='if specified, use CPU only')

    arg_list = deepcopy(sys.argv)
    args = parser.parse_args()

    if args.arch is None or args.nonlin is None or args.lr is None:
        print('ERROR: arch, nonlin, and lr arguments must be specified\n')
        parser.print_help()
        exit(-1)

    uses_tp = (args.nonlin == 'sign11' or args.nonlin == 'qrelu')

    curtime = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    arch_str = args.arch
    op_str = args.opt + '_lr{}_mu{}_wd{}{}'.format(args.lr, args.momentum, args.wtdecay, 'noval' if args.no_val else '')
    tp_str = args.nonlin + ('-' + args.tp_rule if uses_tp else '')
    args.save = os.path.join('logs', args.ds, curtime + '.' + arch_str + '.' + args.loss + '.' + op_str + '.' + tp_str)

    if args.test_model:
        args.save = args.save + '_test'
        args.no_val = True

    args.tp_rule = tp.TPRule[args.tp_rule]

    gpu_str = ','.join(str(g) for g in args.gpus)
    if not args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        print('using cuda device{}: {}'.format('s' if len(args.gpus) > 1 else '', gpu_str))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    log_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)-5.5s] %(message)s',
                                      datefmt='%Y.%m.%d %H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    print("logging to file '{}.log'".format(args.save))
    file_handler = logging.FileHandler(args.save + '.log')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)

    logging.info('command line call: {}'.format(" ".join(arg_list)))
    logging.info('arguments: {}'.format(args))

    if not args.test_model:
        logging.info("training a deep network with the specified arguments")
    else:
        logging.info("testing model '{}' with the specified arguments".format(args.test_model))

    backcompat.broadcast_warning.enabled = True
    warnings.filterwarnings("ignore", "volatile was removed and now has no effect")

    # ----- create datasets -----
    train_loader, val_loader, test_loader, num_classes, embedding_vector = \
        create_datasets(args.ds, args.batch, args.no_val, args.cuda, seed)

    metrics = {'loss': Metric('loss', float('inf'), False),
               'acc': Metric('acc', 0.0, True)}
    metrics = {'train': deepcopy(metrics), 'val': deepcopy(metrics), 'test': deepcopy(metrics)}

    # ----- create loss function -----
    loss_function = get_loss_function(args.loss)

    # either train a model from scratch or load and evaluate a model from disk
    if not args.test_model:
        # ----- create model -----
        model = create_model(args, num_classes, embedding_vector)

        logging.info('created {} model:\n {}'.format(arch_str, model))
        logging.info("{} model has {} parameters".format(arch_str,
                                                         sum([p.data.nelement() for p in model.parameters()])))
        print('num params: ', {n: p.data.nelement() for n, p in model.named_parameters()})

        # ----- create optimizer -----
        optimizer = get_optimizer(model, args)

        if val_loader:
            logging.info('evaluating training on validation data (train size = {}, val size = {}, test size = {})'
                         .format(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
        else:
            logging.info('not using validation data (train size = {}, test size = {})'
                         .format(len(train_loader.dataset), len(test_loader.dataset)))

        cudnn.benchmark = True

        # ----- train the model -----
        timers = {}
        best_model_state = train_model(args.epochs, model, optimizer, loss_function, train_loader,
                                       None if args.no_val else val_loader, test_loader, args.cuda,
                                       args.lr_decay, args.lr_decay_epochs, args.save, metrics, args, timers)

        logging.info('testing on trained model ({})'.format('final' if args.no_val else 'best'))
        model.load_state_dict(best_model_state)
        test_model(model, loss_function, test_loader, args.cuda, True)

    else:
        model = create_model(args, num_classes, embedding_vector)
        logging.info("loading test model from '{}'".format(args.test_model))
        state = torch.load(args.test_model)
        model.load_state_dict(state['model_state'])
        test_model(model, loss_function, test_loader, args.cuda, True)

    print('')
    logging.info("log file: '{}.log'".format(args.save))
    logging.info("log dir: '{}'".format(args.save))
    if not args.test_model:
        logging.info("best accuracy model: '{}'".format(os.path.join(args.save, best_model_name)))


def train_model(num_epochs, model, optimizer, loss_function, train_loader, val_loader, test_loader,
                use_cuda, lr_decay, lr_decay_epochs, log_dir, metrics, args, timers):
    epoch = 1
    best_model_state = None
    timer = partial(timer_context, timers_dict=timers)

    if isinstance(lr_decay_epochs, int):
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_decay_epochs, gamma=lr_decay)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_epochs, gamma=lr_decay)

    def train_epoch(epoch):
        model.train()
        ds_size = len(train_loader.dataset)
        num_batches = int(np.ceil(float(ds_size) / train_loader.batch_size))

        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                with timer('cuda'):
                    data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            with timer('forward'):
                optimizer.zero_grad()
                output = model(data)

            with timer('loss'):
                loss = loss_function(output, target)
                lossf = loss.data.squeeze().item()
                output = output.data

            with timer('backward'):
                loss.backward()

            with timer('optimizer'):
                optimizer.step()

            with timer('output'):
                top_pred = output.max(dim=1)[1]
                acc = 100.0 * top_pred.eq(target.data).float().cpu().mean().item()
                assert target.dim() == 1

                metrics['train']['loss'].update(lossf, epoch)
                metrics['train']['acc'].update(acc, epoch)

            with timer('output'):
                if num_batches <= 5 or (batch_idx % (num_batches // 5)) == 0:
                    logging.info("Train epoch {} [{}/{} ({:.0f}%)]:\t loss = {:.6f}, accuracy = {:.2f}"
                                 .format(epoch, (batch_idx + 1) * len(data), ds_size,
                                         100. * (batch_idx + 1) / len(train_loader), lossf, acc))

    def test(epoch, data_loader, is_val):
        ds_size = len(data_loader.dataset)
        test_loss, acc = test_model(model, loss_function, data_loader, use_cuda, False)
        with timer('output'):
            if is_val:
                print('')
            logging.info('{} set: average loss = {:.4f}, accuracy = {}/{} ({:.2f}%)'
                         .format('Validation' if is_val else 'Test', test_loss,
                                 round(acc * ds_size / 100), ds_size, acc))
            print('')

        ds_name = 'val' if is_val else 'test'
        metrics[ds_name]['loss'].update(test_loss, epoch)
        is_best = metrics[ds_name]['acc'].update(acc, epoch)
        nonlocal best_model_state
        if is_val and is_best:
            best_model_state = deepcopy(model.state_dict())

        return is_best

    try:
        # run the train + test loop for <num_epochs> iterations
        for epoch in range(1, num_epochs + 1):
            is_best = False
            lr_scheduler.step()

            with timer('train'):
                train_epoch(epoch)

            if val_loader is not None:
                with timer('val'):
                    is_best = test(epoch, val_loader, True)

            if test_loader is not None:
                with timer('test'):
                    test(epoch, test_loader, False)

            if epoch % 10 == 0 or epoch == num_epochs:
                logging.info('timings: {}'.format(', '.join('{}: {:.3f}s'.format(*tt) for tt in
                                                            zip(timers.keys(), timers.values()))))

            checkpoint_model(epoch, model=model, opt=optimizer, args=args, log_dir=log_dir,
                             metrics=metrics, timers=timers, is_best=is_best)

    except KeyboardInterrupt:
        print('KeyboardInterrupt: shutdown requested ... exiting')
        sys.exit(0)

    finally:
        for ds_name in ['train'] + (['val'] if val_loader is not None else []) \
                       + (['test'] if test_loader is not None else []):
            logging.info('best {} accuracy: {:.2f}% occurred on epoch {} / {}'
                         .format(ds_name, metrics[ds_name]['acc'].val,
                                 metrics[ds_name]['acc'].tag, epoch))

        logging.info('timings: {}'.format(', '.join('{}: {:.3f}s'.format(*tt) for tt in
                                                    zip(timers.keys(), timers.values()))))

    # if no validation set, then just use the final model
    if best_model_state is None:
        best_model_state = model.state_dict()

    return best_model_state


def test_model(model, loss_function, data_loader, use_cuda, log_results=True):
    model.eval()
    loss, num_correct, nsamples = 0, 0, len(data_loader.dataset)
    with torch.no_grad():
        for data, target in data_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            batch_loss = torch.squeeze(loss_function(output, target)).item()
            loss += batch_loss * data.size(0)
            output, target = output.data, target.data
            num_correct += output.max(1)[1].eq(target).float().cpu().sum().item()
    test_acc = 100. * num_correct / nsamples
    loss /= nsamples

    if log_results:
        log_str = 'Test set: average loss = {:.4f}, accuracy = {}/{} ({:.2f}%)' \
            .format(loss, num_correct, nsamples, test_acc)
        logging.info(log_str)

    return loss, test_acc


def get_loss_function(loss_str):
    if loss_str == 'crossent':
        loss_function = F.cross_entropy
    else:
        raise NotImplementedError('no other loss functions currently implemented')
    return loss_function


def get_optimizer(model, args):
    parameters = filter(lambda x: x.requires_grad, model.parameters())
    if args.opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.wtdecay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=args.lr, weight_decay=args.wtdecay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.wtdecay)
    else:
        raise NotImplementedError('no other optimizers currently implemented')
    return optimizer


def create_model(args, num_classes, embedding_vector):
    nl_str = args.nonlin.lower()
    if nl_str == 'relu':
        nonlin = nn.ReLU
    elif nl_str == 'threshrelu':
        nonlin = ThresholdReLU
    elif nl_str == 'sign11':
        nonlin = partial(Sign11, targetprop_rule=args.tp_rule)
    elif nl_str == 'qrelu':
        nonlin = partial(qReLU, targetprop_rule=args.tp_rule, nsteps=3)
    else:
        raise NotImplementedError('no other non-linearities currently supported')

    # input size
    if args.ds == 'sentiment140' or args.ds == 'tsad':
        input_shape = (1, 60, 50)
    else:
        raise NotImplementedError('no other datasets currently supported')

    # create a model with the specified architecture
    if args.arch == 'cnn':
        model = CNN(input_shape, num_classes, embedding_vector, nonlin=nonlin)
    elif args.arch == 'lstm':
        model = LSTM(input_shape, num_classes, embedding_vector)
    elif args.arch == 'cnn-lstm':
        model = CNN_LSTM(input_shape, num_classes, embedding_vector, nonlin=nonlin)
    elif args.arch == 'lstm-cnn':
        model = LSTM_CNN(input_shape, num_classes, embedding_vector, nonlin=nonlin)
    elif args.arch == 'textcnn':
        model = TextCNN(input_shape, num_classes, embedding_vector, nonlin=nonlin)
    else:
        raise NotImplementedError('other models not yet supported')

    logging.info("{} model has {} parameters and non-linearity={} ({})"
                 .format(args.arch, sum([p.data.nelement() for p in model.parameters()]),
                         nl_str, args.tp_rule.name))

    if len(args.gpus) > 1:
        model = nn.DataParallel(model)

    if args.cuda:
        model.cuda()

    return model


def checkpoint_model(epoch, model, opt, args, log_dir, metrics, timers, is_best):
    if not log_dir:
        return

    state = {
        'epoch': epoch,
        'arch': args.arch,
        'nonlin': args.nonlin,
        'loss': args.loss,
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'metric_state': metrics,
        'args': args,
        'timers': timers,
    }

    # checkpoint full training state
    cp_name = 'model_checkpoint_epoch{}.pth.tar'
    best_file = os.path.join(log_dir, best_model_name)
    torch.save(state, os.path.join(log_dir, cp_name.format(epoch)))
    if epoch > 1:
        prev_cp = os.path.join(log_dir, cp_name.format(epoch - 1))
        if os.path.exists(prev_cp) and os.path.isfile(prev_cp):
            os.remove(prev_cp)
    else:
        logging.info("model checkpoints will be saved to file '{}' after each epoch".format(cp_name.format(epoch)))
        logging.info("the best accuracy model will be saved to '{}'".format(best_file))

    # save model state for best model
    del state['opt_state']
    if is_best:
        torch.save(state, best_file)


if __name__ == '__main__':
    main()
