import os
import numpy as np
import tensorflow as tf
from child_network import Child_ConvNet
from cifar10_processor import get_tf_dataset_from_numpy


def exp_moving_avg(rewards):

    """
    Description:
        We use this as the baseline
        function for our REINFORCE gradient calculation, as mentioned previously, to
        calculate the exponential moving average of the past rewards:

    Arguments:
        rewards  - list of rewards
        
    Returns:
        The last value of exponential moving average
    """

    weights = np.exp(np.linspace(-1. , 0. , len(rewards)))
    weights /= sum(weights)
    a = np.convolve(rewards, weights, mode="full")[:len(rewards)]
    return a[-1]


class Controller(object):

    def __init__(self):

        self.graph = tf.Graph()
        self.sess = tf.Session(graph= self.graph)
        self.num_cell_outputs = controller_params['components_per_layer']* controller_params['max_layers'] # from config

        self.reward_history = []
        self.architecture_history = []
        self.division_rate = 100

        with self.graph.as_default():
            self.build_controller() # method for building controller
 

