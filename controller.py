import os
import numpy as np
import tensorflow as tf
from child_network import Child_ConvNet
from cifar10_processor import get_tf_dataset_from_numpy
from config import controller_params


def exp_moving_avg(rewards):

    """
    Description:
        We use this as the baseline
        function for our REINFORCE gradient calculation to
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
        """
        Description:
            Contstructor that initializes our controller which
            will be a reccurent neural network
            
            graph - tensorflow graph
            sess - tensorflow session
            cell outputs - based on a config file, number of values that our RNN will output - corresponds to child network architecture
            reward history - history of rewards
            architecture history - history of analyzed and train cnn childs

        Arguments:
            self
            
        Returns:
            Nothing, initializes object.
        """
        self.graph = tf.Graph()
        self.sess = tf.Session(graph= self.graph)
        self.num_cell_outputs = controller_params['components_per_layer']* controller_params['max_layers'] # from config

        self.reward_history = []
        self.architecture_history = []
        self.division_rate = 100

        with self.graph.as_default():
            self.build_controller() # method for building controller

    def build_controller(self):
        """
        Description:

            This method builds up our controller - Reccurent Neural Network

        Arguments:
            self
            
        Returns:
            controller
        """      
        with tf.name_scope('controller_inputs'):

            # Input to NAS cell
            self.child_network_architecture = tf.placeholder(tf.float32, [None, self.num_cell_outputs], name='controller_input')

            # Discounted rewards
            self.cnn_dna_output = tf.placeholder(tf.float32, [None, self.num_cell_outputs], name = 'discounted_rewards')

        
        print('Building  controller network')

        with tf.name_scope('network_generation'):
            with tf.name_scope('controller'):

                self.controller_output = tf.identity(self.network_generator(self.child_network_architecture), name='policy_scores')

                # We cast our output tensor to int because the list of numbers that rnn outputs must be int - look at child network 
                self.cnn_dna_output = tf.cast(tf.scalar_mul(self.division_rate, self.controller_output), tf.int32, name='controller_prediction')
        
        print('Model is built')
        print('Setting up the optimizer')
        # Now we set up our optimizer and its behaviour during training
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.99, self.global_step, 500, 0.96, staircase=True)
        self.optimizer = tf.train.Adam(learning_rate = self.learning_rate)
        print('Optimizer setup finished')

        print('Calculating gradients and loss')

        with tf.name_scope('gradient_and_loss'):

            self.policy_gradient_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(

                logits = self.controller_output[:,-1,:],
                labels = self.child_network_architecture
            ))

            # We will use l2 regularization method for preventing overfitting of controller weights

            self.l2_loss = tf.reduce_sum(tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables(scope='controller')]))
            
            # Now we will addd the to losses (with regularization parameter beta) to define total loss
            self.total_loss = self.policy_gradient_loss + self.l2_loss * controller_params["beta"]

            # Compute the gradients of the network over the loss
            self.gradients = self.optimizer.compute_gradients(self.total_loss)

            print('Computing REINFORCE gradients')
            # Compute gradients using REINFORCE 
            for i, (grad, var) in enumerate(self.gradients):

                if grad is not None:
                    self.gradients[i] = (grad*self.discounted_rewards, var)

            print('Applying REINFORCE gradients to controller')
            
            with tf.name_scope('train_controller'):
                self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

            print('Finished building a controller')





 

