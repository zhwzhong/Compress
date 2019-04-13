# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   test_lstm.py
@Time    :   2018/11/14 22:31
@Desc    :
"""
import math
import os
import time

import numpy as np
import torch as t
import torchnet as tnt
from matplotlib import pyplot as plt
from torch.utils import data

import arithmeticcoding
from LSTM import opt
from datasets import Dataset
from model import GenRNN

static_freqs = {'F': 5976988, 'E': 1315670, 'G': 997280, 'D': 547197, 'C': 330023, 'B': 262461, 'A': 217139,
                '@': 179314,
                '?': 145585, '>': 136129, '=': 111259, '<': 98971, ';': 86561, ':': 76098, '9': 67100, '8': 55670,
                '7': 48703, '6': 42709, '5': 35666, '4': 31135, '3': 26870, '2': 23120, '1': 22869, '0': 19068,
                '/': 17308, '.': 15466, ',': 13562, '-': 13240, "'": 11860, '+': 11694, '(': 11050, ')': 10927,
                '*': 10913,
                '&': 10657, '%': 2038}

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_device_number
characters = ['%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
              '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def generate_freqs(pro, first_step=False, resolution=1e9):
    freqs = arithmeticcoding.SimpleFrequencyTable([0] * (1 + len(characters)))
    for i in range(len(characters)):
        # freqs.set(i, static_freqs[characters[i]])
        freqs.set(i, 1)
    if first_step is False:
        for i in range(pro.shape[0]):
            if (pro[i] * resolution).astype(np.int64) > 1:
                freqs.set(i, (pro[i] * resolution).astype(np.int64))
    freqs.set(len(characters), 1)  # \n
    # freqs.set(41, 1)  # EOF
    return freqs


def predict(model, context, hidden=None):
    device = t.device(opt.device)
    context = t.from_numpy((context.reshape(1, -1, 1))).type(t.FloatTensor).to(device)
    output, hidden = model(context, hidden)
    output = t.nn.functional.softmax(output, dim=1)
    output_numpy = output.cpu().detach().numpy().astype(np.float32)
    return output_numpy, hidden


def str_to_int(string):
    return np.array([char2int[i] for i in list(string)]).reshape(1, opt.chunk_len, 1)


def compress(model):
    bit_out = arithmeticcoding.BitOutputStream(open('./result/data.bin', "wb"))
    enc = arithmeticcoding.ArithmeticEncoder(bit_out)
    # model = GenRNN(input_size=1, hidden_size=opt.hidden_size, output_size=len(characters), n_layers=opt.num_layers)
    # device = t.device(opt.device)
    # model.load_state_dict(t.load('./checkpoints/net_{}.pth'.format(opt.model_name, opt.chunk_len)))
    z = open('./result/old.txt', 'w')
    # model = model.to(device)
    # model.eval()
    hidden = None
    num_line = 0
    sum_all = 0
    time_num = 0
    acc_num = 0
    end_freq = generate_freqs(pro=1, first_step=True)
    with open('./result/test.qs') as f:
        while True:
            text = f.readline().replace('\n', '')
            z.write(text)
            z.write('\n')
            if not text:
                break
            encode_text = [char2int[char] for char in text]
            num_line += 1
            hidden = None
            for char_index in range(len(encode_text)):
                if char_index == 0:
                    freq = generate_freqs(pro=1, first_step=True)
                    sum_all += -np.log2(1 / 35.)
                    time_num += 8.0
                    # enc.write(freq, encode_text[char_index])
                else:
                    target_char = np.array(encode_text[char_index])
                    context_char = np.array(encode_text[char_index - 1])
                    out, hidden = predict(model, context_char, hidden)
                    out = out[0]  # (35, )
                    sum_all += -np.log2(out[target_char])
                    time_num += 8.0
                    freq = generate_freqs(pro=out, first_step=False)

                    if np.argmax(out) == target_char.astype(np.int):
                        acc_num += 1
                enc.write(freq, encode_text[char_index])
                end_freq = freq
            # enc.write(end_freq, 40)

            if num_line % 100 == 0:
                print(num_line)
            if num_line > 10000:
                break
    freq = generate_freqs(pro=1, first_step=True)
    # print(end_freq)
    enc.write(end_freq, len(characters))
    enc.finish()
    print(acc_num / time_num, sum_all / time_num)


def decompress(model):
    dec_char = ''
    bit_in = arithmeticcoding.BitInputStream(open('./result/data.bin', 'rb'))
    dec = arithmeticcoding.ArithmeticDecoder(bit_in)
    out_f = open('./result/recover.txt', 'w')

    index = 0
    num_line = 0
    hidden = None
    # context = []
    while True:

        if index == 0:
            freq = generate_freqs(pro=1, first_step=True)
            dec_char = dec.read(freq)
            index += 1
            # print(freq.frequencies, end='')
            # print(int2char[dec_char])
            if dec_char == len(characters):
                break
            # context.append(int2char[dec_char])
            out_f.write(int2char[dec_char])
        else:
            out, hidden = predict(model, np.array(dec_char), hidden)
            out = out[0]  # (35, )
            freq = generate_freqs(pro=out, first_step=False)
            index += 1
            dec_char = dec.read(freq)
            if dec_char == len(characters):
                break
            # context.append(int2char[dec_char])
            out_f.write(int2char[dec_char])
        if index == 100:
            index = 0
            num_line += 1
            hidden = None
            out_f.write('\n')
            if num_line % 100 == 0:
                print(num_line)
            # context = []
    out_f.close()


def test():
    test_data = Dataset(chunk_len=opt.chunk_len, train=False)

    test_loader = data.DataLoader(dataset=test_data, batch_size=opt.test_batch_size,
                                  shuffle=True, num_workers=16, drop_last=True)

    model = GenRNN(input_size=1, hidden_size=opt.hidden_size, output_size=len(characters), n_layers=opt.num_layers)
    device = t.device(opt.device)
    # 上一层目录有备份
    model.load_state_dict(t.load('./checkpoints/net_{}.pth'.format(opt.model_name, opt.chunk_len)))

    model = model.to(device)
    model.eval()
    start_time = time.time()
    test_loss_meter = tnt.meter.AverageValueMeter()
    test_accuracy_meter = tnt.meter.AverageValueMeter()
    criterion = t.nn.CrossEntropyLoss().to(device)
    sum_all = 0
    time_num = 0
    print("Total batch for test: {}".format(len(test_loader)))
    hidden = None
    sum_pb = 0

    sum_pb_list = []
    for step, test_data in enumerate(test_loader):

        batch_all = 0
        batch_num = 0

        hidden = None
        test_x = test_data[:, :-1].unsqueeze(2)
        test_y = t.squeeze(test_data[:, 1:])

        first_chr = test_x[:, 0, :].detach().cpu().numpy()

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
        # 10982300
        for chr_i in range(first_chr.shape[0]):
            sum_all += np.log2(static_freqs[int2char[first_chr[chr_i][0]]] / 10982300.0)
            batch_all += np.log2(static_freqs[int2char[first_chr[chr_i][0]]] / 10982300.0)
            time_num += 1
            batch_num += 1
        # print output_numpy[10].sum(), "------"
        # sum_all += np.log2(1 / 35.) * opt.test_batch_size
        # time_num += opt.test_batch_size
        for i in range(target_numpy.shape[0]):
            sum_all += np.log2(output_numpy[i][target_numpy[i]])
            sum_pb += output_numpy[i][target_numpy[i]]

            batch_all += np.log2(output_numpy[i][target_numpy[i]])
            batch_num += 1

            sum_pb_list.append(output_numpy[i][target_numpy[i]])
            time_num += 1
        # print(output_numpy.shape)
        # input("Please input")
        pred = np.argmax(output_numpy, axis=1)
        accuracy = float((pred == target_numpy).astype(int).sum()) / float(target_numpy.size)
        test_accuracy_meter.add(accuracy)
        if step % 1000 == 0:
            batch_bpc = -1 * (batch_all / batch_num)
            print("Step: {}, Time {}, Acc: {}, Bpc: {}".format(step, time_since(start_time),
                                                               test_accuracy_meter.value()[0], batch_bpc / 8.))
    bpc = -1 * (sum_all / time_num)
    print("TestLoss: {}, TestAccuracy: {}, TestBpc: {}, TestTime: {}. TotalChars: {}, Compression Ratio: {}".
          format(test_loss_meter.value()[0], test_accuracy_meter.value()[0], bpc, time_since(start_time), time_num,
                 bpc / 8.))
    print("MEAN_Pro", sum_pb / time_num)
    plt.hist(sum_pb_list, bins=35)
    plt.show()


if __name__ == '__main__':
    # test()
    model = GenRNN(input_size=1, hidden_size=opt.hidden_size, output_size=len(characters), n_layers=opt.num_layers)
    device = t.device(opt.device)
    # model.load_state_dict(t.load('./checkpoints/net_{}.pth'.format(opt.model_name, opt.chunk_len), map_location='cpu'))
    model.load_state_dict(t.load('./checkpoints/net_{}.pth'.format(opt.model_name, opt.chunk_len)))
    model = model.to(device)
    model.eval()
    print("Loading Model...")
    start_time = time.time()
    compress(model)
    print("Encode Time", time.time() - start_time)

    start_time = time.time()
    decompress(model)
    print("Decode Time", time.time() - start_time)
