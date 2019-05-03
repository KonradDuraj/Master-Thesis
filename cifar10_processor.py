import os
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import np_utils




"""
        Description:
            This script processes CIFAR-10 dataset which will be used to 
            train the cnn child networks. To be more precise we create a pipeline
            for input data using tf.data.Dataset API - this method is much more 
            efficient when it comes to training neural networks than the traditional
            way via: sess.run and feed_dict ={}

"""

def _create_tf_dataset(x, y, batch_size):
    """
    Description:
        private method
        Creates tf.data.Dataset object.

    Arguments:
        x,y - data and labels
        batch_size
        
    Returns:
        tf.data.Dataset object
    """

    return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y))).shuffle(500).repeat().batch(batch_size)

def get_tf_dataset_from_numpy(batch_size, validation_split = 0.1):

        """
        Description:
            Main function getting tf.Data.datasets for training, validation, and testing.
            The data is downloaded via keras dataset API

        Arguments:
            batch_size
            validation_split - split for partitioning training and validation sets.

        Returns:
            train, valid, test - sets
            number of batches for: train, valid, test sets
        """
        (X, y), (X_test, y_test) = cifar10.load_data()
        print('Data loaded from keras')

        X = X / 255.
        X_test = X_test / 255.

        X = X.astype(np.float32)
        X_test = X_test.astype(np.float32)

        y = y.astype(np.float32)
        y_test = y_test.astype(np.float32)

        if y.shape[1] !=10:
            y = np_utils.to_categorical(y, num_classes=10)
            y_test = np_utils.to_categorical(y_test, num_classes=10)

        print('Data preprocessed')

        split_idx = int((1.0 - validation_split) * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_valid, y_valid = X[split_idx:], y[split_idx:]

        train_dataset = _create_tf_dataset(X_train, y_train, batch_size)
        valid_dataset = _create_tf_dataset(X_valid, y_valid, batch_size)
        test_dataset = _create_tf_dataset(X_test, y_test, batch_size)

        print('Created validation dataset')

        # Get the batch sizes for the train, valid, and test datasets
        num_train_batches = int(X_train.shape[0] // batch_size)
        num_valid_batches = int(X_valid.shape[0] // batch_size)
        num_test_batches = int(X_test.shape[0] // batch_size)

        print('Batches are calculated')

        return train_dataset, valid_dataset, test_dataset, num_train_batches, num_valid_batches, num_test_batches