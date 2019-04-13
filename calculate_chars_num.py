# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   calculate_chars_num.py
@Time    :   2018/11/20 08:24
@Desc    :
"""
import numpy as np
characters = ['%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
              '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

# int2char = dict(enumerate(characters))
# # print(int2char)
# char2int = {char: index for index, char in int2char.items()}

characters = [ord(char) for char in characters]
print(characters)
print(int("01111111", 2))
print(int('10000000', 2))
low_list = []
high_list = []
for i in characters:
    high = i & 127
    low = i & 128
    low_list.append(low)
    high_list.append(high)
print("high length", len(set(low_list)))
print("low length", len(set(high_list)))
# encoded_text = np.array([(char2int[char] & 248) >> 3 for char in text]).reshape(-1, 1)
# new_chars = [(char2int[char] & 248) >> 3 for char in characters]
# characters = sorted(list(set(new_chars)))