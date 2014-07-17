#!/usr/bin/env python

import theano
print theano.config.device

train = open('/home/vagrant/pylearn2/pylearn2/scripts/tutorials/convolutional_network/conv.yaml', 'r').read()
train_params = {'train_stop': 100,
                    'valid_stop': 50100,
                    'test_stop': 500,
                    'batch_size': 100,
                    'output_channels_h2': 64, 
                    'output_channels_h3': 64,  
                    'max_epochs': 500,
                    'save_path': '.'}
train = train % (train_params)
print train



from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()
