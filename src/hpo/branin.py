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
from pylearn2.models.mlp import ConvElemWise
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorihtms.learning_rule import Momemtum
from pylearn2.training_algorihtms.learning_rule import AdaDelta
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.cost import MethodCost
from pylearn2.costs.mlp import WeightDecay
from pylearn2.termination_criterion import And
from pylearn2.termination_criterion import MonitorBased
from pylearn2.termination_criterion import EpochCounter
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.learning_rule import MomemtumAdjustor

import time


def buildDataset(training_set_size, params, **kwargs):
    """
    This function is responsible for building the MNIST dataset object. It accepts an integer which controls how large
    the training set is.
    """
    d = MNIST(which_set='train', one_hot=1, start=0, stop=training_set_size)
    return d


def buildModel(params, **kwargs):
    """
    This function is responsible for building the model object. The input and output topology is fixed.
    -- Parameters --
    batch_size: integer
    remaining arguments are used by other functions.
    """
    conv_2d_space = Conv2DSpace(shape=[28, 28], num_channels=1)

    m = MLP(batch_size=kwargs["mlp_batch_size"], input_space=conv_2d_space, layers=buildLayers(params, **kwargs))
    return m


def buildLayers(params, **kwargs):
    """
    This function is responsible for building a list of layer objects.
    """
    layers = []
    for layer_num in range(1, kwargs["num_conv_layers"]):
        if kwargs[layer_num + "_conv_type"] == "ConvRectifiedLinear":
            new_layer = ConvRectifiedLinear(layer_name=kwargs[layer_num + "_layer_name"],
                                            output_channels=kwargs[layer_num + "_output_channels"],
                                            kernel_shape=[kwargs[layer_num + "_kernel_shape_1"],
                                                          kwargs[layer_num + "_kernel_shape_2"]],
                                            pool_shape=[kwargs[layer_num + "_pool_shape"],
                                                        kwargs[layer_num + "_pool_shape"]],
                                            irange=kwargs[layer_num + "_irange"],
                                            border_mode='valid',
                                            include_prob=kwargs[layer_num + "_include_prob"],
                                            init_bias=kwargs[layer_num + "_init_bias"],
                                            W_lr_scale=kwargs[layer_num + "_W_lr_scale"],
                                            b_lr_scale=kwargs[layer_num + "_b_lr_scale"],
                                            max_kernel_norm=kwargs[layer_num + "_max_kernel_norm"],
                                            pool_type='max',
                                            tied_b=kwargs[layer_num + "_tied_b"],
                                            kernel_stride=[kwargs[layer_num + "_kernel_stride_1"],
                                                           kwargs[layer_num + "_kernel_stride_2"]],
                                            monitor_style='classification')
        elif kwargs[layer_num + "_conv_type"] == "ConvElemWise":
            new_layer = ConvRectifiedLinear(layer_name=kwargs[layer_num + "_layer_name"],
                                            output_channels=kwargs[layer_num + "_output_channels"],
                                            kernel_shape=[kwargs[layer_num + "_kernel_shape_1"],
                                                          kwargs[layer_num + "_kernel_shape_2"]],
                                            pool_shape=[kwargs[layer_num + "_pool_shape"],
                                                        kwargs[layer_num + "_pool_shape"]],
                                            irange=kwargs[layer_num + "_irange"],
                                            border_mode='valid',
                                            include_prob=kwargs[layer_num + "_include_prob"],
                                            init_bias=kwargs[layer_num + "_init_bias"],
                                            W_lr_scale=kwargs[layer_num + "_W_lr_scale"],
                                            b_lr_scale=kwargs[layer_num + "_b_lr_scale"],
                                            max_kernel_norm=kwargs[layer_num + "_max_kernel_norm"],
                                            pool_type='max',
                                            tied_b=kwargs[layer_num + "_tied_b"],
                                            kernel_stride=[kwargs[layer_num + "_kernel_stride_1"],
                                                           kwargs[layer_num + "_kernel_stride_2"]],
                                            monitor_style='classification',
                                            nonlinearity=kwargs[layer_num + "_nonlinearity"])
        layers.append(new_layer)
    return layers


def buildAlgorithm(params, **kwargs):
    """
    This function is responsible for building an algorithm object. An algorithm consists of a few parameters,
    a monitoring dataset (which will be constant), a cost function (which for now will also be constant),
    and a termination_criterion (yet another constant).
    """

    m_dataset = {'valid': MNIST(
        which_set='train',
        one_hot=1,
        start=50000,
        stop=60000
    ),
                 'test': MNIST(
                     which_set='test',
                     one_hot=1,
                     stop=10000
                 )}
    cost = SumOfCosts(costs=[MethodCost(
        method='cost_from_X'
    ), WeightDecay(
        coeffs=[.00005, .00005, .00005]
    )])

    termination_criterion = And(
        criteria=[MonitorBased(
            channel_name="valid_y_misclass",
            prop_decrease=0.50,
            N=10
        ),
                  EpochCounter(
                      max_epochs=kwargs["max_epochs"]
                  )])

    algorithm = SGD(batch_size=kwargs["batch_size"],
                    learning_rate=kwargs["learning_rate"],
                    learning_rule=buildLearningRule(params, kwargs),
                    monitoring_dataset=m_dataset,
                    cost=cost,
                    termination_criterion=termination_criterion)
    return algorithm


def buildExtensions(params, **kwargs):
    extensions = []
    extensions.append(MonitorBasedSaveBest(
        channel_name='test_y_misclass',
        save_path="./convolutional_network_best.pkl"))
    if kwargs["learning_rule"] == "Momemtum":
        extensions.append(MomentumAdjustor(
            start=1,
            saturate=10,
            final_momentum=.99
        ))
    return extensions


def main(params, **kwargs):
    train = Train(dataset=buildDataset(params, **kwargs),
                  model=buildModel(params, **kwargs),
                  algorithm=buildAlgorithm(params, **kwargs),
                  extensions=buildExtensions(params, **kwargs))
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
