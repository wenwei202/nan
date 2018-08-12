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
import numpy as np
import sys
import matplotlib.pyplot as plt

current_module = sys.modules[__name__]

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch NNaNUnit test')

parser.add_argument('--dims', default='10,10,10',
                    help='sizes of hidden neurons - e.g. 5,8,10')
parser.add_argument('--nonlinear', default='relu',
                    help='the nonlinear function to approximate - e.g. tanh ')
parser.add_argument('--range', default='-20,20',
                    help='the domain to approximate')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size')
parser.add_argument('--iterations', type=int, default=5000,
                    help='steps of SGD')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')


def adjust_lr(num_iter, max_iter, base_lr, power=0.5):
    return base_lr * ((1.0 - num_iter/max_iter) ** power)


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def tanh(x):
    return np.tanh(x)


def exp(x):
    return np.exp(x)


def asqrt(x):
    return np.sqrt(abs(x))


def abs(x):
    return np.abs(x)


def cos(x):
    return np.cos(x)


def sin(x):
    return np.sin(x)


def main():
    args = parser.parse_args()
    args.dims = [int(i) for i in args.dims.split(',')]
    args.range = [int(i) for i in args.range.split(',')]
    args.nonlinear = getattr(current_module, args.nonlinear)
    torch.cuda.set_device(0)

    model = models.__dict__['nnan_unit'](dims=args.dims)
    criterion = nn.MSELoss(size_average=True, reduce=True)
    criterion.type(args.type)
    model.type(args.type)
    func = args.nonlinear
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for num_iter in range(args.iterations):
        if num_iter % 500 == 0:
            optimizer = torch.optim.SGD(model.parameters(),
                                        weight_decay=0.0005,
                                        lr=adjust_lr(num_iter, args.iterations, args.lr))
        optimizer.zero_grad()
        # generate a random batch
        inputs = torch.rand(args.batch_size, 2, 3) * (args.range[1] - args.range[0]) + args.range[0]
        targets = func(inputs)
        targets = targets.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=False)
        target_var = Variable(targets)
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()

        if num_iter % 100 == 0:
            print('loss: {}'.format(loss.data[0]))

    # plot the learned function
    xs = np.linspace(2*args.range[0], 2*args.range[1], 1000)
    ts = func(xs)
    input_var = torch.from_numpy(xs)
    input_var = Variable(input_var.type(args.type), volatile=True)
    output = model(input_var)
    ys = output.data.cpu().numpy()
    plt.plot(xs, ts, 'b-', label='expected')
    plt.plot(xs, ys, 'r--', label='learned')
    plt.legend()
    plt.title('Comparison')
    plt.show()


if __name__ == '__main__':
    main()
