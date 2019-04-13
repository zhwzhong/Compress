# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   model.py
@Time    :   2018/11/14 19:30
@Desc    :
"""
import torch as t


class GenRNN(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(GenRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_size
        self.rnn = t.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.dropout = t.nn.Dropout(p=0.5)
        self.bn = t.nn.BatchNorm1d(num_features=hidden_size)
        self.fc = t.nn.Linear(hidden_size, output_size)

    def forward(self, inputs, h_state=None):
        batch_size, seq_len, feature = inputs.shape
        # 初始化隐藏状态
        if h_state is None:
            # print(self.n_layers, batch_size, self.hidden_dim)
            h_0 = inputs.data.new(self.n_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = inputs.data.new(self.n_layers, batch_size, self.hidden_dim).fill_(0).float()
            h_state = (h_0, c_0)

        output, h_state = self.rnn(inputs, h_state)

        # output = self.fc2(self.dropout(self.fc1(self.dropout(output.contiguous().view(batch_size * seq_len, -1)))))
        # output = self.bn(output.contiguous().view(batch_size * seq_len, -1))
        output = self.fc(output.contiguous().view(batch_size * seq_len, -1))
        return output, h_state
