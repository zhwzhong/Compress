# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   mo_test.py
@Time    :   2018/11/16 17:10
@Desc    :
"""
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
#
# # torch.manual_seed(1)    # reproducible
#
# # make fake data
# n_data = torch.ones(100, 2)
# x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
# y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
# x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
# y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
# y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
#
# # The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# # x, y = Variable(x), Variable(y)
#
# # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# # plt.show()
#
#
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.out = torch.nn.Linear(n_hidden, n_output)   # output layer
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.out(x)
#         return x
#
# net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
# print(net)  # net architecture
#
# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
#
# plt.ion()   # something about plotting
#
# for t in range(100):
#     out = net(x)                 # input x and predict based on x
#
#     loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
#     print(x.size(), y.size(), x.dtype, y.dtype, type(x), type(y))
#     break
#     optimizer.zero_grad()   # clear gradients for next train
#     loss.backward()         # backpropagation, compute gradients
#     optimizer.step()        # apply gradients
#
#     if t % 2 == 0:
#         # plot and show learning process
#         plt.cla()
#         prediction = torch.max(out, 1)[1]
#         pred_y = prediction.data.numpy()
#         target_y = y.data.numpy()
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
#         plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()

static_freqs = {'F': 5976988, 'E': 1315670, 'G': 997280, 'D': 547197, 'C': 330023, 'B': 262461, 'A': 217139,
                '@': 179314,
                '?': 145585, '>': 136129, '=': 111259, '<': 98971, ';': 86561, ':': 76098, '9': 67100, '8': 55670,
                '7': 48703, '6': 42709, '5': 35666, '4': 31135, '3': 26870, '2': 23120, '1': 22869, '0': 19068,
                '/': 17308, '.': 15466, ',': 13562, '-': 13240, "'": 11860, '+': 11694, '(': 11050, ')': 10927,
                '*': 10913,
                '&': 10657, '%': 2038}

import numpy as np


def get_pro(chr):
    sum_freq = 0
    for key, value in static_freqs.items():
        sum_freq += value
    print sum_freq
    return np.float32(static_freqs[chr]) / sum_freq


print get_pro('F')