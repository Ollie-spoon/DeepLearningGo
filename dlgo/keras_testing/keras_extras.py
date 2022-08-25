"""
The purpose of this file is to provide some additional resources that do
not fit within the standard multi layer perceptron MLP mnist example.
"""

import tensorflow.keras as keras  # ???
import keras.backend as backend


def myactivation(x):
    """
    This function is a custom activation function taking 3 inputs and passing

    :param x: The output of the previous layer.
    :return:
    """

    #each element in the last dimension is a neuron
    "This might be modifiable to be just 0, 1, and 2. "
    "I don't know if theres a reason for that"
    n0 = x[:, 0:1]
    n1 = x[:, 1:2]
    n2 = x[:, 2:3]  #each N is shaped as (batch_size, 1)

    #apply the activation to each neuron
    x0 = backend.relu(n0)
    x1 = backend.tanh(n1)
    x2 = backend.sigmoid(n2)

    return backend.concatenate([x0, x1, x2], axis=-1)
