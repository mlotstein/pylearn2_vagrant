#!/usr/bin/env python

import theano
import os
import sys
import time
import gc
import pickle
print theano.config.device

train = open('./conv.yaml', 'r').read()
train_params = {'train_stop': 500,
                    'valid_stop': 50500,
                    'test_stop': 10000,
                    'batch_size': 100,
                    'output_channels_h2': 8, 
                    'output_channels_h3': 8,  
                    'max_epochs': 10,
                    'save_path': '.'}
train = train % (train_params)
print train



from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()
model = pickle.load(open('best_model.pkl', 'rb'))
monitor = model.monitor
channels = monitor.channels
def read_channel(s):
	return float(channels[s].val_record[-1])
print 'job#, orig valid, valid both, new test, old test'
v, t = map(read_channel, ['valid_y_misclass', 'test_y_misclass'])
print t