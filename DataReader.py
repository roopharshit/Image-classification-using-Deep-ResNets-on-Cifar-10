import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """

    x_train, y_train = [], []
    x_test, y_test = [], []
    #Load train data
    for i in range(1, 6):
        file_path = os.path.join(data_dir, 'data_batch_{}'.format(i))
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            x_train.append(batch[b'data'])
            y_train.append(batch[b'labels'])
    
    #Load test data
    file_path = os.path.join(data_dir, 'test_batch')
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x_test = batch[b'data']
        y_test = batch[b'labels']
    
    #convert to numpy and normalize
    x_train = np.concatenate(x_train).astype('float')/255
    y_train = np.concatenate(y_train).astype('int32')
    x_test = np.array(x_test).astype('float32') / 255
    y_test = np.array(y_test).astype('int32')


    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid