# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   LSTM.py
@Time    :   2018/11/14 19:19
@Desc    :
"""
import os
import math
import time
import shutil
import argparse
import torch as t
import numpy as np
import torchnet as tnt
from model import GenRNN
from torch.utils import data
from datasets import Dataset
from utils import time_since, create_dir
from gen_data import data_process
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='LSTM FOR GEN DATA PROCESS')

parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--RNN', type=str, default='lstm')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--chunk_len', type=int, default=99)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--reload', type=bool, default=False)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--replace', type=bool, default=True)
parser.add_argument('--test_batch_size', type=int, default=128)
# chunk_len_replace
parser.add_argument('--model_name', type=str, default='LSTM')
parser.add_argument('--val_batch_size', type=int, default=128)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--cuda_device_number', type=str, default='0')
parser.add_argument('--data_attr', type=str, default='')
parser.add_argument('--epoch', type=int, default=20)

# Chunk_len 最大只能为99

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_device_number
# characters = ['\n', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
#               '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

characters = ['%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
              '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G']


int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}

print(opt)


def val(model, val_loader, device, train_step, writer):
    model.eval()
    start_time = time.time()
    test_loss_meter = tnt.meter.AverageValueMeter()
    test_accuracy_meter = tnt.meter.AverageValueMeter()
    criterion = t.nn.CrossEntropyLoss().to(device)
    sum_all = 0
    time_num = 0
    print("Total batch for test: {}".format(len(val_loader)))
    hidden = None
    for step, test_data in enumerate(val_loader):
        test_x = test_data[:, :-1].unsqueeze(2)
        test_y = t.squeeze(test_data[:, 1:])
        test_x = test_x.type(t.FloatTensor).to(device)
        test_y = test_y.type(t.LongTensor).to(device)

        output, hidden = model(test_x, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())

        loss = criterion(output, test_y.contiguous().view(-1))
        test_loss_meter.add(loss.item())
        # 准确率和熵的计算
        output = t.nn.functional.softmax(output, dim=1)
        output_numpy = output.cpu().detach().numpy().astype(np.float32)

        target_numpy = test_y.contiguous().view(-1).cpu().detach().numpy()

        for i in range(target_numpy.shape[0]):
            sum_all += np.log2(output_numpy[i][target_numpy[i]])
            time_num += 1
        pred = np.argmax(output_numpy, axis=1)
        accuracy = float((pred == target_numpy).astype(int).sum()) / float(target_numpy.size)
        test_accuracy_meter.add(accuracy)
        if step % 1000 == 0:
            print("Step: {}, Time {}, Acc: {}".format(step, time_since(start_time), test_accuracy_meter.value()[0]))
    bpc = -1 * (sum_all / time_num)
    print("TestLoss: {}, TestAccuracy: {}, TestBpc: {}, TestTime: {}. TotalChars: {}".
          format(test_loss_meter.value()[0], test_accuracy_meter.value()[0], bpc, time_since(start_time), time_num))
    writer.add_scalar("TestLoss", test_loss_meter.value()[0], train_step)
    writer.add_scalar("TestAccuracy", test_accuracy_meter.value()[0], train_step)
    writer.add_scalar("TestBpc", bpc, train_step, train_step)
    writer.add_scalar("TestCompressionRatio", bpc / 8, train_step)
    model.train()


def train():
    train_data = Dataset(chunk_len=opt.chunk_len, train=True)

    val_data = Dataset(chunk_len=opt.chunk_len, train=False)

    train_loader = data.DataLoader(dataset=train_data, batch_size=opt.train_batch_size,
                                   shuffle=False, num_workers=1, drop_last=True)
    val_loader = data.DataLoader(dataset=val_data, batch_size=opt.train_batch_size,
                                 shuffle=False, num_workers=1, drop_last=True)

    model = GenRNN(input_size=1, hidden_size=opt.hidden_size, output_size=len(characters), n_layers=opt.num_layers)

    device = t.device(opt.device)
    if opt.reload:
        model.load_state_dict(t.load('./data/checkpoints/net_{}.pth'.format(opt.model_name)))

    model = model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = t.nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter('./logs/{}'.format(opt.model_name))
    train_loss_meter = tnt.meter.AverageValueMeter()
    accuracy_meter = tnt.meter.AverageValueMeter()
    step = 0
    start_time = time.time()
    for epoch in range(opt.epoch):
        # print('Epoch', epoch)
        print(len(train_loader))
        for _, train_data in enumerate(train_loader):
            hidden = None
            step += 1
            model.zero_grad()
            # train_X (batch, seq_len, feature), train_y: (batch, seq_len)

            train_x = train_data[:, :-1].unsqueeze(2)
            train_y = t.squeeze(train_data[:, 1:])
            train_x = train_x.type(t.FloatTensor).to(device)
            train_y = train_y.type(t.LongTensor).to(device)
            # print(train_x.shape, train_y.shape)
            output, hidden = model(train_x, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            loss = criterion(output, train_y.contiguous().view(-1))
            train_loss_meter.add(loss.item())
            loss.backward()
            optimizer.step()
            output = t.nn.functional.softmax(output, dim=1)
            output_numpy = output.cpu().detach().numpy()
            target_numpy = train_y.contiguous().view(-1).cpu().detach().numpy()
            pred = np.argmax(output_numpy, axis=1)
            accuracy = float((pred == target_numpy).astype(int).sum()) / float(target_numpy.size)
            accuracy_meter.add(accuracy)
            if step % 1000 == 0:
                sum_all = 0
                time_num = 0
                for i in range(target_numpy.shape[0]):
                    sum_all += np.log2(output_numpy[i][target_numpy[i]])
                    time_num += 1
                bpc = -1 * (sum_all / time_num)
                print('Epoch: {}, Step: {}, Loss: {}, accuracy: {}, Time: {}, TrainBpc: {}, CompRatio: {}'
                      .format(epoch, step, train_loss_meter.value()[0], accuracy, time_since(start_time), bpc, bpc / 8))

                writer.add_scalar("TrainLoss", train_loss_meter.value()[0], step)
                writer.add_scalar("TrainAccuracy", accuracy_meter.value()[0], step)
                writer.add_scalar("TrainBpc", bpc, step)
                writer.add_scalar("TrainCompRatio", bpc / 8, step)
                train_loss_meter.reset()
                accuracy_meter.reset()
            if step % 10000 == 0:
                val(model=model, val_loader=val_loader, device=device, train_step=step, writer=writer)

        t.save(model.state_dict(), './checkpoints/net_{}.pth'.format(opt.model_name))


def test():
    # train_set = np.
    pass


if __name__ == '__main__':
    create_dir('./logs/{}'.format(opt.model_name))
    train()






