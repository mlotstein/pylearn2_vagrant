# #
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import HPOlib.benchmark_util as benchmark_util
import HPOlib.benchmark_functions as benchmark_functions
import theano
from pylearn2.config import yaml_parse
import os
import sys
import time
import gc

def main(params, **kwargs):
    train = open('./branin.yaml', 'r').read()
    train_params = {'train_stop': 500,
					'valid_stop': 50500,
					'test_stop': 10000,
					'batch_size': 100,
					'output_channels': params['output_channels'],
					'irange': params['irange'],
					'kernel_shape': params['kernel_shape'],
					'pool_shape': params['pool_shape'],
					'pool_stride': params['pool_stride'],
					'max_kernel_norm': params['max_kernel_norm'],
					'output_channels_2': params['output_channels_2'],
					'irange_2': params['irange_2'],
					'kernel_shape_2': params['kernel_shape_2'],
					'pool_shape_2': params['pool_shape_2'],
					'pool_stride_2': params['pool_stride_2'],
					'max_kernel_norm_2': params['max_kernel_norm_2'],
					'max_col_norm': params['max_col_norm'],
					'learning_rate': params['learning_rate'],
					'init_momentum': params['init_momentum'],
					'max_epochs': 500,
					'save_path': '.'}
    train = train % (train_params)
    train.main_loop()
    model = serial.load(os.path.join(d, f, 'best_model.pkl'))
    monitor = model.monitor
    channels = monitor.channels
    def read_channel(s):
	return float(channels[s].val_record[-1])
	#print 'job#, orig valid, valid both, new test, old test'
    v, t = map(read_channel, ['valid_y_misclass', 'test_y_misclass'])
    return t

if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
          ("SAT", abs(duration), result, -1, str(__file__))
