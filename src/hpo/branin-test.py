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
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import ConvRectifiedLinear
from pylearn2.models.mlp import ConvElemwise
from pylearn2.models.mlp import RectifierConvNonlinearity
from pylearn2.models.mlp import SigmoidConvNonlinearity
from pylearn2.models.mlp import TanhConvNonlinearity
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.training_algorithms.learning_rule import AdaDelta
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.cost import MethodCost
from pylearn2.costs.mlp import WeightDecay
from pylearn2.termination_criteria import And
from pylearn2.termination_criteria import MonitorBased
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor
from pylearn2.train import Train

import time


def buildDataset():
    """
    This function is responsible for building the MNIST dataset object. It accepts an integer which controls how large
    the training set is.
    """
    d = MNIST(which_set='train', one_hot=1, start=0, stop=1000)
    return d


def buildModel():
    """
    This function is responsible for building the model object. The input and output topology is fixed.
    -- Parameters --
    batch_size: integer
    remaining arguments are used by other functions.
    """
    conv_2d_space = Conv2DSpace(shape=[28, 28], num_channels=1)

    m = MLP(batch_size=kwargs["mlp_batch_size"], input_space=conv_2d_space, layers=buildLayers(params, **kwargs))
    return m


def buildLayers():
    """
    This function is responsible for building a list of layer objects.
    """
    layers = []
    # for layer_num in range(1, 2):
        # if kwargs[layer_num + "_conv_type"] == "ConvRectifiedLinear":
    new_layer_1 = ConvRectifiedLinear(layer_name="l1",
		    output_channels=8,
		    kernel_shape=[4,4],
		    pool_shape=[4,4],
		    irange=.05,
		    border_mode='valid',
		    include_prob=.05,
		    init_bias=0,
		    W_lr_scale=.5,
		    b_lr_scale=.01,
		    max_kernel_norm=1.9,
		    pool_type='max',
		    tied_b=0,
		    kernel_stride=[2,2],
		    monitor_style='classification')
    layers.append(new_layer_1)
        # elif kwargs[layer_num + "_conv_type"] == "ConvElemWise":
    new_layer_2 = ConvRectifiedLinear(layer_name='l2',
		    output_channels=8,
		    kernel_shape=[4,4],
		    pool_shape=[4,4],
		    irange=.05,
		    border_mode='valid',
		    include_prob=.05,
		    init_bias=0.0,
		    W_lr_scale=.5,
		    b_lr_scale=.01,
		    max_kernel_norm=1.9,
		    pool_type='max',
		    tied_b=0,
		    kernel_stride=[2,2],
		    monitor_style='classification',
		    nonlinearity=buildNonlinearity('Rect'))
    layers.append(new_layer_2)
    return layers

def buildNonlinearity(nonlinearity):
	if nonlinearity == 'Rect':
		return RectifierConvNonlinearity()
	elif nonlinearity == 'Sigmoid':
		return SigmoidConvNonlinearity()
	elif nonlinearity == 'Tanh':
		return TanhConvNonlinearity()

def buildAlgorithm():
    """
    This function is responsible for building an algorithm object. An algorithm consists of a few parameters,
    a monitoring dataset (which will be constant), a cost function (which for now will also be constant),
    and a termination_criterion (yet another constant).
    """

    m_dataset = {'valid': MNIST(
        which_set='train',
        one_hot=1,
        start=50000,
        stop=50500
    ),
                 'test': MNIST(
                     which_set='test',
                     one_hot=1,
                     stop=1000
                 )}
    cost = SumOfCosts(costs=[MethodCost(
        method='cost_from_X'
    ), WeightDecay(
        coeffs=[.00005, .00005, .00005]
    )])

    terminatia_criteria = And(
        criteria=[MonitorBased(
            channel_name="valid_y_misclass",
            prop_decrease=0.50,
            N=10
        ),
                  EpochCounter(
                      max_epochs=kwargs["max_epochs"]
                  )])

    algorithm = SGD(batch_size=100,
                    learning_rate=.05,
                    learning_rule=buildLearningRule(params, **kwargs),
                    monitoring_dataset=m_dataset,
                    cost=cost,
                    termination_criteria=termination_criteria)
    return algorithm


def buildExtensions():
    extensions = []
    extensions.append(MonitorBasedSaveBest(
        channel_name='test_y_misclass',
        save_path="./convolutional_network_best.pkl"))
    # if kwargs["learning_rule"] == "Momemtum":
        # extensions.append(MomentumAdjustor(
            # start=1,
            # saturate=10,
            # final_momentum=.99
        # ))
    return extensions


def main(params, **kwargs):
    train = Train(dataset=buildDataset(),
                  model=buildModel(),
                  algorithm=buildAlgorithm(),
                  extensions=buildExtensions())
    train.main_loop()

    result = 0  # here we need to read the file from disk
    # print 'Params: ', params,
    #y = benchmark_functions.save_branin(params, **kwargs)
    #print 'Result: ', y
    return y


if __name__ == "__main__":
    starttime = time.time()

args, params = benchmark_util.parse_cli()
result = main(params, **args)
duration = time.time() - starttime
print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
	  ("SAT", abs(duration), result, -1, str(__file__))
