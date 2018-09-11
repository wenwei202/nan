import pdb
import argparse
import os
import time
import logging
from random import uniform
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from ast import literal_eval
from torch.nn.utils import clip_grad_norm
from math import ceil
from models.resnet import set_nonlinear
import numpy as np
import functools

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR',
                    default='./TrainingResults', help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int,
                    metavar='N', help='mini-batch size (default: 2048)')
parser.add_argument('-mb', '--mini-batch-size', default=128, type=int,
                    help='mini-mini-batch size (default: 128)')
parser.add_argument('--lr_bb_fix', dest='lr_bb_fix', action='store_true',
                    help='learning rate fix for big batch lr =  lr0*(batch_size*batch_multiplier/128)**0.5')
parser.add_argument('--no-lr_bb_fix', dest='lr_bb_fix', action='store_false',
                    help='learning rate fix for big batch lr =  lr0*(batch_size*batch_multiplier/128)**0.5')
parser.set_defaults(lr_bb_fix=False)
parser.add_argument('--save_all', dest='save_all', action='store_true',
                    help='save all better checkpoints')
parser.add_argument('--no-save_all', dest='save_all', action='store_false',
                    help='save all better checkpoints')
parser.set_defaults(save_all=False)
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='data augment')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='data augment')
parser.set_defaults(augment=True)
parser.add_argument('--regime_bb_fix', dest='regime_bb_fix', action='store_true',
                    help='regime fix for big batch e = e0*(batch_size*batch_multiplier/128)')
parser.add_argument('--no-regime_bb_fix', dest='regime_bb_fix', action='store_false',
                    help='regime fix for big batch e = e0*(batch_size*batch_multiplier/128)')
parser.set_defaults(regime_bb_fix=False)
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=None, type=float,
                    metavar='W', help='weight decay (default: None)')
parser.add_argument('--dropout', default=None, type=float,
                    metavar='DROPOUT', help='dropout ratio (default: None)')
parser.add_argument('--sharpness-smoothing', '--ss', default=0.0, type=float,
                    metavar='SS', help='sharpness smoothing (default: 0)')
parser.add_argument('--anneal-index', '--ai', default=0.55, type=float,
                    metavar='AI', help='Annealing index of noise (default: 0.55)')
parser.add_argument('--tanh-scale', '--ts', default=10., type=float,
                    metavar='TS', help='scale of transition in tanh')
parser.add_argument('--smoothing-type', default='constant', type=str, metavar='ST',
                    help='The type of chaning smoothing noise: constant, anneal or tanh')
parser.add_argument('--adapt-type', default='none', type=str, metavar='AT',
                    help='The type of adapting noise: none, weight or filter')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--batch-multiplier', '-bm', default=1, type=int,
                    metavar='BM', help='The number of batchs to delay parameter updating (default: 1). Used for very large-batch training using limited memory')
parser.add_argument('--dims', default=None,
                    help='sizes of hidden neurons - e.g. 5,8,10')
parser.add_argument('--depth', default=44, type=int,
                    metavar='N', help='depth of resent (default: 44)')

def main():
    #torch.manual_seed(123)
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    if args.dims:
        args.dims = [int(i) for i in args.dims.split(',')]

        class _NonLinear(models.nnan.NNaNUnit):
            def __init__(self, **kwds):
                super(_NonLinear, self).__init__(dims=args.dims, **kwds)

        set_nonlinear(_NonLinear)
    if args.regime_bb_fix:
            args.epochs *= (int)(ceil(args.batch_size*args.batch_multiplier / args.mini_batch_size))

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
      raise OSError('Directory {%s} exists. Use a new one.' % save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.info("run arguments: %s", args)

    if 'cuda' in args.type:
        #torch.cuda.manual_seed_all(123)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'depth': args.depth}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=args.augment),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    if args.optimizer == 'Adam':
      assert(args.weight_decay is not None)
      regime = {0: {'optimizer': args.optimizer,
                'lr': args.lr,
                'weight_decay': args.weight_decay}}
    else:
      regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
      if args.weight_decay:
          regime[0]['weight_decay'] = args.weight_decay
    adapted_regime = {}
    for e, v in regime.items():
        if args.lr_bb_fix and 'lr' in v:
            v['lr'] *= (args.batch_size*args.batch_multiplier / args.mini_batch_size) ** 0.5
        if args.regime_bb_fix:
            e *= ceil(args.batch_size*args.batch_multiplier / args.mini_batch_size)
        adapted_regime[e] = v
    regime = adapted_regime


    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)
    print({i: list(w.size())
           for (i, w) in enumerate(list(model.parameters()))})
    init_weights = [w.data.cpu().clone() for w in list(model.parameters())]

    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_result = train(train_loader, model, criterion, epoch, optimizer)

        train_loss, train_prec1, train_prec5 = [
            train_result[r].data.tolist() for r in ['loss', 'prec1', 'prec5']]

        # evaluate on validation set
        val_result = validate(val_loader, model, criterion, epoch)
        val_loss, val_prec1, val_prec5 = [val_result[r].data.tolist()
                                          for r in ['loss', 'prec1', 'prec5']]

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        if is_best:
            logging.info('\n Epoch: {0}\t'
                         'Best Val Prec@1 {val_prec1:.3f} '
                         'with Val Prec@5 {val_prec5:.3f} \n'
                         .format(epoch + 1, val_prec1=val_prec1, val_prec5=val_prec5))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path, save_all=args.save_all)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        #Enable to measure more layers
        idxs = [0]#,2,4,6,7,8,9,10]#[0, 12, 45, 63]

        step_dist_epoch = {'step_dist_n%s' % k: (w.data.cpu() - init_weights[k]).norm().data.tolist()
                           for (k, w) in enumerate(list(model.parameters())) if k in idxs}


        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5,
                    **step_dist_epoch)

        results.plot(x='epoch', y=['train_loss', 'val_loss'],
                     title='Loss', ylabel='loss')
        results.plot(x='epoch', y=['train_error1', 'val_error1'],
                     title='Error@1', ylabel='error %')
        results.plot(x='epoch', y=['train_error5', 'val_error5'],
                     title='Error@5', ylabel='error %')

        for k in idxs:
            results.plot(x='epoch', y=['step_dist_n%s' % k],
                         title='step distance per epoch %s' % k,
                         ylabel='val')

        results.save()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    if training:
        optimizer.zero_grad()

    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        # compute output
        if not training:
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data[0], input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

        else:
            is_updating = ((i+1)%args.batch_multiplier == 0) or (i+1==len(data_loader))
            mini_inputs = input_var.chunk(args.batch_size // args.mini_batch_size)
            mini_targets = target_var.chunk(args.batch_size // args.mini_batch_size)


            # get the coefficent to scale noise
            eq_batch_num = (len(data_loader)+args.batch_multiplier-1)//args.batch_multiplier
            if args.smoothing_type == 'constant':
              noise_coef = 1.
            elif args.smoothing_type == 'anneal':
              noise_coef = 1.0 / ((1 + epoch * eq_batch_num + i//args.batch_multiplier) ** args.anneal_index)
              noise_coef = noise_coef ** 0.5
            elif args.smoothing_type == 'tanh':
              noise_coef = np.tanh(args.tanh_scale*((float)(epoch * eq_batch_num + i//args.batch_multiplier)/(float)(args.epochs * eq_batch_num) -.5))
              noise_coef = (noise_coef + 1.)/2.0
            else: raise ValueError('Unknown smoothing-type')
            if i % args.print_freq == 0:
              logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                           'Noise Coefficient: {noise_coef:.4f}\t'.format(
                epoch, i, len(data_loader),
                phase='TRAINING' if training else 'EVALUATING',
                noise_coef=noise_coef))

            for k, mini_input_var in enumerate(mini_inputs):

                noises = {}
                # randomly change current model @ each mini-mini-batch
                if args.sharpness_smoothing:
                    for key, p in model.named_parameters():
                      if hasattr(model, 'quiet_parameters') and (key in model.quiet_parameters):
                          continue

                      if args.adapt_type == 'weight':
                        noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.) * args.sharpness_smoothing * torch.abs(p.data) * noise_coef

                      elif args.adapt_type == 'filter':
                        noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.)
                        noise_shape = noise.shape
                        noise_norms = noise.view([noise_shape[0],-1]).norm(p=2, dim=1) + 1.0e-6
                        p_norms = p.view([noise_shape[0], -1]).norm(p=2, dim=1)
                        for shape_idx in range(1, len(noise_shape)):
                            noise_norms = noise_norms.unsqueeze(-1)
                            p_norms = p_norms.unsqueeze(-1)
                        noise = noise / noise_norms * p_norms.data
                        #for idx in range(0, noise.shape[0]):
                        #  if 1 == len(noise.shape):
                        #    if np.abs(np.linalg.norm(noise[idx]))>1.0e-6:
                        #      noise[idx] = noise[idx] / np.linalg.norm(noise[idx]) * np.linalg.norm(p.data[idx])
                        #  else:
                        #    if np.abs(noise[idx].norm())>1.0e-6:
                        #      noise[idx] = noise[idx] / noise[idx].norm() * p.data[idx].norm()

                        noise = noise * args.sharpness_smoothing * noise_coef

                      elif args.adapt_type == 'none':
                        noise = (torch.cuda.FloatTensor(p.size()).uniform_() * 2. - 1.) * args.sharpness_smoothing * noise_coef

                      else:
                          raise ValueError('Unkown --adapt-type')
                      noises[key] = noise
                      p.data.add_(noise)

                mini_target_var = mini_targets[k]
                output = model(mini_input_var)
                loss = criterion(output, mini_target_var)

                prec1, prec5 = accuracy(output.data, mini_target_var.data, topk=(1, 5))
                losses.update(loss.data[0], mini_input_var.size(0))
                top1.update(prec1[0], mini_input_var.size(0))
                top5.update(prec5[0], mini_input_var.size(0))

                # compute gradient and do SGD step
                loss.backward()

                # denoise @ each mini-mini-batch.
                if args.sharpness_smoothing:
                    for key, p in model.named_parameters():
                      if key in noises:
                        p.data.sub_(noises[key])

            if is_updating:
              n_batches = args.batch_multiplier
              if (i+1) == len(data_loader):
                  n_batches = (i % args.batch_multiplier) + 1
              for p in model.parameters():
                  p.grad.data.div_(len(mini_inputs)*n_batches)
              clip_grad_norm(model.parameters(), 5.)
              optimizer.step()
              optimizer.zero_grad()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return {'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg}


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
