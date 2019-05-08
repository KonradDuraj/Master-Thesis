# -*- coding: utf-8 -*-

import tensorflow as tf
from config import LOGS_DIR
import os


class Child_ConvNet(object):
    
    def __init__(self, cnn_dna, child_id, num_of_classes, beta_l2=1e-3, dropout_rate=0.3, **kwargs):
        
        """
        Description:
            Constructor of the child network, that initialize 
            starter parameters for networks that will be generated
            
        Arguments:
            cnn_dna - list with specific information about number of convolutional layers and their parameters
            For example given a list that looks like:
                
                [3,2,2,3, 1,2,3,4]
                
            We can say that our network has 2 convolutional layers where the first layer has:
                - kernel size = 3
                - stride = 2
                - filters = 2
                - max pool window size = 3
                
           num_of_classes - number of classes for a given problem
           child id - string which helps identify the network
           beta l2 - weight parameter for L2 regularization
           dropout rate - parameter passed into dropout layer, specifying the number of neurons to be deactivated
          
        Returns:
            Nothing, only sets the parameters for the given network.
        """
        
        self.cnn_dna = self.process_raw_controller_output(cnn_dna)
        self.child_id = child_id
        self.num_of_classes = num_of_classes
        self.beta_l2 = beta_l2
        self.dropout_rate = dropout_rate
        
        # placeholder_with_default - a placeholder op that passes through input when its output is not fed.
        self.is_training = tf.placeholder_with_default(True, shape=None, name='is_training')
        
    
    def process_raw_controller_output(self, output):
        
        """
        Description:
            This method parses the cnn_dna that the controller outputs
            
        Arguments:
            output - output of the controller - NAS cell
            
        Returns:
            The child network architecture in a form of list of lists
            [ [1,3,4,2],
              [3,4,2,1],
              .
              .
              .
            ]
        """
        
        output = output.ravel()
        cnn_dna = [list(output[x:x+4]) for x in range(0, len(output), 4)]
        
        return cnn_dna
        
    def build_network(self, input_tensor):
        
        """
        Description:
            This method creates the networks using the cnn_dna parameter
            
        Arguments:
            input_tensor - tensor representing input data; shape of this data
            for example for CIFAR-10 that would be: (number of samples, 32,32,3)
            
        Returns:
            
            The tensor that represents the output logit (pre-softmax activation)
        """       
        log_file = open(os.path.join(LOGS_DIR, 'child_logger.txt'), 'a+')
        log_file.write(f'\t DNA for the network is: {self.cnn_dna}\t')
        
        output=input_tensor
        
        for index in range(0, len(self.cnn_dna)):
            
            # get config parameters for the layers
            print(self.cnn_dna[index])
            kernel_size, stride, num_of_filters, max_pool_size = self.cnn_dna[index]

            scp_name =  f'child_{self.child_id}/convlayer_{index}' #"child_{}_conv_layer_{}".format(str(self.child_id), str(index))
            with tf.name_scope(scp_name):
                
                output = tf.layers.conv2d(inputs=output,
                                          kernel_size=(kernel_size,kernel_size),
                                          filters=num_of_filters,
                                          strides=(stride, stride),
                                          padding='SAME',
                                          name=f'conv_layer{index}',
                                          
                                          # Fixed activation function - changeable
                                          activation=tf.nn.relu,
                                          
                                          # Xavier initializer: 
                                          #https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.zeros_initializer(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.beta_l2))
                
                # Now that we've created the conv layer we will hardcode
                # the maxpool layers - this can be changed in the future
                
                output = tf.layers.max_pooling2d(output, 
                                                 pool_size=(max_pool_size, max_pool_size),
                                                 strides=1, 
                                                 padding='SAME',
                                                 name=f'max_pool_{index}')
                
                # Now the dropout layers
                
                output = tf.layers.dropout(output, rate=self.dropout_rate, training=self.is_training)
                
                # Now we flatten our inputs and pass it through our
                # dense layer which also is hardcoded
                
                with tf.name_scope(f'child_{self.child_id}_fully_connected_layer'):
                    
                    output = tf.layers.flatten(output, name='flatten')
                    logits = tf.layers.dense(output, self.num_of_classes, name='dense')

                    log_file.close()    
                    
                return logits
            
            
    
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        