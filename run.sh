#!/usr/bin/env bash

python LSTM.py --chunk_len 99 --model_name '99_true_no_rela_small' --cuda_device_number '0' --replace True --data_attr 'small' --epoch 200 &
python LSTM.py --chunk_len 99 --model_name '99_true_no_rela_big' --cuda_device_number '0' --replace True --epoch 20 &

python LSTM.py --chunk_len 99 --model_name '99_false_no_rela_small' --cuda_device_number '2' --replace False --data_attr 'small' --epoch 200 &



# python LSTM.py --chunk_len 10 --model_name '10_true_no_rela_small' --cuda_device_number '1' --replace True --data_attr 'small' --epoch 200