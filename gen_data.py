# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   gen_data.py
@Time    :   2018/11/14 19:20
@Desc    :
"""
import numpy as np

file_10m_len = 102400
characters = ['%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
              '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}


def data_process(file_name, replace=False):
    """
    读取fastq文件，保存成训练所需要的格式
    :return:
    """
    with open(file_name) as tmp_f:
        line_len = len(tmp_f.readline())
    print("Line_len", line_len)
    text = open(file_name, 'r').read()
    text.replace('\n', '')
    encoded_text = [char2int[char] for char in text]

    np.savez_compressed(file_name.replace('.qs', '_{}.npz'),
                        data=np.array(encoded_text).reshape(-1, 1))
    print("Data write to npz file")


def fastq_to_qs(file_name):
    qs = open(file_name.replace('fastq', 'qs'), 'w')
    with open(file_name) as fg:
        while True:
            _, _, _, quality_score = fg.readline(), fg.readline(), fg.readline(), fg.readline()
            if not quality_score:
                break
            qs.write(quality_score)
    qs.close()


def get_data(file_name):
    num_lines = 0
    test_num_lines = 0
    fastq = open('./data/train.fastq', 'w')
    test_fastq = open('./data/test.fastq', 'w')
    with open(file_name) as fg:
        while True:
            tmp = [fg.readline(), fg.readline(), fg.readline(), fg.readline()]
            num_lines += 1
            for line in tmp:
                fastq.write(line)
            if len(tmp) == 0:
                break
            if num_lines > file_10m_len * 100:
                break
        while True:
            tmp = [fg.readline(), fg.readline(), fg.readline(), fg.readline()]
            test_num_lines += 1
            for line in tmp:
                test_fastq.write(line)
            if len(tmp) == 0:
                break
            if test_num_lines > file_10m_len * 100:
                break

    fastq.close()


def get_test_data(file_name):
    num_lines = 0
    train_lines = 0
    fastq = open('./data/test.fastq', 'w')
    with open(file_name) as fg:
        while True:
            tmp = [fg.readline(), fg.readline(), fg.readline(), fg.readline()]
            train_lines += 1
            if train_lines < file_10m_len * 100:
                continue
            if len(tmp) == 0:
                break
            # Sample Data
            rand_n = np.random.randint(0, 100)
            if rand_n > 97:
                num_lines += 1
                for line in tmp:
                    fastq.write(line)
                print(train_lines)

            if num_lines > file_10m_len * 5:
                break
    fastq.close()


if __name__ == '__main__':
    File = '/data/zhwzhong/datasets/Gene/NA12878_read_1.fq'
    # get_data(File)
    # get_test_data(File)
    fastq_to_qs('./data/train.fastq')
    fastq_to_qs('./data/test.fastq')
    # data_process('./data/train.qs', chunk_len=100, replace=True)
