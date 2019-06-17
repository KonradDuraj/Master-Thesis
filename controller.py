import os
import numpy as np
import tensorflow as tf
from child_network import Child_ConvNet
from cifar10_processor import get_tf_dataset_from_numpy
from config import controller_params,child_network_params
import keras.layers as kl
"""
def exp_moving_avg(rewards):

    
    Description:
        We use this as the baseline
        function for our REINFORCE gradient calculation to
        calculate the exponential moving average of the past rewards:

    Arguments:
        rewards  - list of rewards
        
    Returns:
        The last value of exponential moving average
    

    weights = np.exp(np.linspace(-1. , 0. , len(rewards)))
    weights /= sum(weights)
    a = np.convolve(rewards, weights, mode="full")[:len(rewards)]
    return a[-1]
"""

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
        self.clip_norm = 0.0

        with self.graph.as_default():
            self.build_controller() # method for building controller

    def network_generator(self, nas_cell_hidden_state): 
    
        """
        Description:

            This method initializes our full controller model

        Arguments:
            nas_cell_hidden_state - state of the nas cell
            
        Returns:
            controller
        """  
        # number of output units we expect from NAS cell

        with tf.name_scope('network_generation'):
            
            nas = tf.contrib.rnn.NASCell(self.num_cell_outputs)

            #network_architecture, nas_cell_hidden_state = tf.nn.dynamic_rnn(nas, tf.expand_dims(nas_cell_hidden_state, -1), dtype=tf.float32)
            network_architecture, nas_cell_hidden_state = tf.nn.dynamic_rnn(nas, tf.expand_dims(nas_cell_hidden_state, -1), dtype=tf.float32)

            bias_variable = tf.Variable([0.05]* self.num_cell_outputs)
            network_architecture = tf.nn.bias_add(network_architecture, bias_variable)

            print('Network architecture ', network_architecture)
            print('Returned architecture: ', network_architecture[:,-1:,:])
            return network_architecture[:,-1:,:]

    def generate_child_network(self, child_network_architecture):
        """
        Description:

            This method generates child cnn networks

        Arguments:
            Architecture of cnn child - described via list of lists of parameters for each layer
            
        Returns:
            Parsed and initialized cnn child
        """  
        with self.graph.as_default():
            return self.sess.run(self.cnn_dna_output, {self.child_network_architectures:child_network_architecture})

    def build_controller(self):
        """
        Description:

            This method builds up our controller - Reccurent Neural Network

        Arguments:
        Returns:
            Nothing, only sets the controller
        """      
        with tf.name_scope('controller_inputs'):

            # Input to NAS cell
            self.child_network_architectures = tf.placeholder(tf.float32, [None, self.num_cell_outputs], name='controller_input') 

            # Discounted rewards
            self.discounted_rewards  = tf.placeholder(tf.float32, (None,), name = 'discounted_rewards')

        
        print('Building  controller network')

        with tf.name_scope('network_generation'):
            with tf.variable_scope('controller'):

                self.controller_output = tf.identity(self.network_generator(self.child_network_architectures), name='policy_scores')

                # We cast our output tensor to int because the list of numbers that rnn outputs must be int - look at child network 
                self.cnn_dna_output = tf.cast(tf.scalar_mul(self.division_rate, self.controller_output), tf.int32, name='controller_prediction')
        
        print('Model is built')
        print('Setting up the optimizer')
        # Now we set up our optimizer and its behaviour during training
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.05, self.global_step, 50,0.96, staircase=True)
        #self.learning_rate = 0.3
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
        print('Optimizer setup finished')

        print('Calculating gradients and loss')

        with tf.name_scope('gradient_and_loss'):

            self.policy_gradient_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(

                logits = self.controller_output[:,-1,:],
                labels = self.child_network_architectures
            ))

            policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="controller")

            # We will use l2 regularization method for preventing overfitting of controller weights
            self.l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
            
            # Now we will addd the to losses (with regularization parameter beta) to define total loss
            self.total_loss = self.policy_gradient_loss +   self.l2_loss * controller_params["beta"]

            

            # Compute the gradients of the network over the loss
            self.gradients = self.optimizer.compute_gradients(self.total_loss) # ,tf.GraphKeys.TRAINABLE_VARIABLES

            with tf.name_scope('policy_gradients'):
                    # normalize gradients so that they dont explode if argument passed
                    if self.clip_norm is not None and self.clip_norm != 0.0:
                        norm = tf.constant(self.clip_norm, dtype=tf.float32)
                        gradients, vars = zip(*self.gradients)  # unpack the two lists of gradients and the variables
                        gradients, _ = tf.clip_by_global_norm(gradients, norm)  # clip by the norm
                        self.gradients = list(zip(gradients, vars))  # we need to set values later, convert to list


            print('Computing REINFORCE gradients')
            # Compute gradients using REINFORCE 
            for i, (grad, var) in enumerate(self.gradients):

                if grad is not None:
                    self.gradients[i] = (grad*self.discounted_rewards, var)

            print('Applying REINFORCE gradients to controller')
            
            with tf.name_scope('train_controller'):
                self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

            print('Finished building a controller')

    def train_child_network(self, cnn_dna, child_id):

        """
        Description:

            This method train the child cnn network

        Arguments:
            cnn_dna - list of lists describing cnn child architecture
            child_cnn - id for our child network
            
        Returns:
            validation accuracy
        """  

        child_graph = tf.Graph()
        with child_graph.as_default():
            
            

            sess = tf.Session()
            child_network = Child_ConvNet(cnn_dna=cnn_dna,child_id=child_id, num_of_classes=10, **child_network_params)

            

            train_dataset, valid_dataset, test_dataset, num_train_batches, num_valid_batches, num_test_batches = get_tf_dataset_from_numpy(batch_size=child_network_params['batch_size'])
            
            # Generic iterator
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            next_tensor_batch = iterator.get_next()

            # Separate train and validation set init ops
            train_init_ops = iterator.make_initializer(train_dataset)
            valid_init_ops = iterator.make_initializer(valid_dataset)

            # Building the graph
            input_tensor, labels = next_tensor_batch

            # Build the child network, which returns the pre-softmax logits of the child network
            logits = child_network.build_network(input_tensor)

            # Define the loss function for child network
            loss_ops = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name='loss')

            # Define the training operation for the child network
            train_ops = tf.train.AdamOptimizer(learning_rate=child_network_params['learning_rate']).minimize(loss_ops)

            # Now we calculate the accuracy of our network

            pred_ops = tf.nn.softmax(logits, name='preds')
            correct = tf.equal(tf.argmax(pred_ops, 1), tf.argmax(labels,1), name='correct')
            accuracy_ops = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

            initializer = tf.global_variables_initializer()

            

            sess.run(initializer)
            sess.run(train_init_ops)

            
            print(f'Training CNN {child_id} for {child_network_params["max_epochs"]} epochs')

            for epoch_idx in range(child_network_params['max_epochs']):

                avg_loss, avg_acc = [], []

                for batch_idx in range(num_train_batches):

                    loss, _, accuracy = sess.run([loss_ops, train_ops, accuracy_ops])
                    avg_loss.append(loss)
                    avg_acc.append(accuracy)
                
                
                print(f'\t Epoch {epoch_idx}: \t loss: {np.mean(avg_loss)} \t accuracy: {np.mean(avg_acc)}')

            print('Validation and returned rewards')
            sess.run(valid_init_ops)
            avg_val_loss, avg_val_acc = [], []
            for batch_idx in range(num_valid_batches):
                valid_loss, valid_accuracy = sess.run([loss_ops, accuracy_ops])
                avg_val_loss.append(valid_loss)
                avg_val_acc.append(valid_accuracy)

            print(f'\tValidation loss: {np.mean(avg_val_loss)}\t accuracy: {np.mean(avg_val_acc)}')

      
        return np.mean(avg_val_acc)

    def train_controller(self):
        """
        Description:
            This method trains the controller 

        Arguments:
            
        Returns:
            Nothing, trains the reccurent network(controller)
        """
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        step = 0
        total_rewards=  0
        child_network_architecture = np.array([[20.0, 15.0, 150.0, 12.0] * controller_params['max_layers']], dtype=np.float32)

        #controller_file = open(os.path.join(LOGS_DIR, 'controller_logger.txt'), 'a+')
        for episode in range(controller_params['max_episodes']):

            #controller_file.write(f'Episode {episode} for controller')
            print(f'\t Episode {episode} for controller \t ')
            step +=1
            episode_reward_buffer = []

            for sub_child in range(controller_params["num_children_per_episode"]):

                # generating child architecture

                child_network_architecture = self.generate_child_network(child_network_architecture)[0]

                if np.any(np.less_equal(child_network_architecture, 0.0)):
                    reward = -1.0
                else:
                    reward = self.train_child_network(cnn_dna=child_network_architecture, child_id = f'child_episode_{episode}_sub_child_{sub_child}')

                episode_reward_buffer.append(reward)

            mean_reward = np.mean(episode_reward_buffer)

            self.reward_history.append(mean_reward)
            self.architecture_history.append(child_network_architecture)

            total_rewards += mean_reward

            child_network_architecture = np.array(self.architecture_history[-step:]).ravel() / self.division_rate
            child_network_architecture = child_network_architecture.reshape((-1, self.num_cell_outputs))

            #baseline = exp_moving_avg(self.reward_history)
            

            print('REWARD HISTORY: ',self.reward_history)
            
            
            last_reward = self.reward_history[-1]
            
            reward = [last_reward]
            print('REWARD: ' , reward)

           

            with self.graph.as_default():

                _, loss = self.sess.run([self.train_op, self.total_loss],
                                        {self.child_network_architectures: child_network_architecture,
                                        self.discounted_rewards: reward})

                

            print(f'Episode: {episode} | Loss: {loss} | DNA: {child_network_architecture.ravel()} | Reward: {mean_reward}')
           
        








 

