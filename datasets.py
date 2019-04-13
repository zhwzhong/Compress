# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   datasets.py
@Time    :   2018/11/15 09:36
@Desc    :
"""
import os

import numpy as np
from torch.utils import data
from torchvision import transforms as T

file_10m_len = 102400

# characters = ['\n', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
#               '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

characters = ['%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
              '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']

int2char = dict(enumerate(characters))
# print(int2char)
char2int = {char: index for index, char in int2char.items()}


def data_process(file_name):
    """
    读取fastq文件，保存成训练所需要的格式
    :return:
    """
    with open(file_name) as tmp_f:
        line_len = len(tmp_f.readline())
    print("Line_len", line_len)
    text = open(file_name, 'r').read()

    text = text.replace('\n', '')
    encoded_text = [char2int[char] for char in text]
    np.savez_compressed(file_name.replace('.qs', '.npz'), data=np.array(encoded_text).reshape(-1, 100))
    print("Data write to npz file")


class Dataset(data.Dataset):
    def __init__(self, chunk_len=99, train=True):

        self.chunk_len = chunk_len
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        if train:

            file_name = './data/train.npz'
            if not os.path.exists(file_name):
                data_process('./data/train.qs')
            self.data = np.load(file_name)['data']
        else:
            file_name = './data/test.npz'
            if not os.path.exists(file_name):
                data_process('./data/test.qs')
            self.data = np.load(file_name)['data']

    def __len__(self):
        return (self.data.shape[0]) * (100 - self.chunk_len)

    def __getitem__(self, item):
        # print(self.data.shape)
        row = item // (100 - self.chunk_len)
        col = item % (100 - self.chunk_len)
        return self.data[row][col: col + self.chunk_len + 1]


if __name__ == '__main__':
    # print(np.load('./data/train_99_true_no_rela.npz')['data'].shape)
    d = Dataset(chunk_len=99)
    d_loader = data.DataLoader(dataset=d, batch_size=128, shuffle=False, num_workers=4)
    print(len(d_loader))
    # for i, x in enumerate(d_loader):
    #     print(x.numpy())
    #     for ii in x.numpy()[0]:
    #         print int2char[ii]
    #     break
    # pass
