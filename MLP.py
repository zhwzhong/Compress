# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   MLP.py
@Time    :   2018/11/14 19:19
@Desc    :
"""
import os
import torch
import numpy as np
import torchnet as tnt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils import data
from tensorboardX import SummaryWriter

characters = ['%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
              '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G']


int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_batch(file_name, seq_len=10):
    data_arr = []
    with open(file_name) as f:
        for i in range(100000):
            text = f.readline().replace('\n', '')
            encode_text = [char2int[char] for char in text]
            for di in range(seq_len, len(encode_text)):
                data_arr.append(encode_text[di - 10: di + 1])
    data_arr = np.array(data_arr)
    np.savez_compressed('./data/mlp.npz', data=data_arr)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


def main():
    train_file = './data/small_train.qs'
    get_batch(train_file, seq_len=10)
    device = torch.device('cuda')
    # device = torch.device('cpu')
    net = Net(n_feature=10, n_hidden=256, n_output=len(characters))
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_data = np.load('./data/mlp.npz')['data']
    train_loader = data.DataLoader(train_data, shuffle=True, batch_size=128, num_workers=2)
    step = 0
    writer = SummaryWriter('./mlp_logs')
    train_loss_meter = tnt.meter.AverageValueMeter()
    accuracy_meter = tnt.meter.AverageValueMeter()
    for epoch in range(3000):

        for i, _x in enumerate(train_loader):
            step += 1
            net.zero_grad()
            train_x, train_y = _x[:, :-1], _x[:, -1]
            train_x = train_x.type(torch.FloatTensor).to(device)
            train_y = train_y.type(torch.LongTensor).to(device)
            output = net(train_x)

            loss = criterion(output, train_y)
            output = torch.nn.functional.softmax(output, dim=1)
            output_numpy = output.cpu().detach().numpy()
            target_numpy = train_y.contiguous().view(-1).cpu().detach().numpy()

            pred = np.argmax(output_numpy, axis=1)
            accuracy = float((pred == target_numpy).astype(int).sum()) / float(target_numpy.size)
            accuracy_meter.add(accuracy)

            train_loss_meter.add(loss.item())
            loss.backward()
            optimizer.step()
            if step % 1000 == 0:
                sum_all = 0
                time_num = 0
                for bpc_i in range(output_numpy.shape[0]):
                    sum_all += -np.log2(output_numpy[bpc_i][target_numpy[bpc_i]])
                    time_num += 1
                bpc = sum_all / time_num
                # print("TrainLoss: ", train_loss_meter.value()[0])
                # print("Accuracy: ", accuracy_meter.value()[0])
                # print("Bpc: ", bpc)
                writer.add_scalar("Bpc", bpc, step)
                writer.add_scalar("Com Ratio", bpc / 8.0, step)
                writer.add_scalar("TrainLoss", train_loss_meter.value()[0], step)
                writer.add_scalar("TrainAccuracy", accuracy_meter.value()[0], step)
                train_loss_meter.reset()
                accuracy_meter.reset()


if __name__ == '__main__':
    main()
    #net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
