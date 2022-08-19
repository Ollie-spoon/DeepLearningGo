## This file contains the code for layers within the nn and
# provides connection, forward pass and backward pass
from abc import ABC

import numpy as np


## Definitions of functions used for relu and sigmoid fns
def sigmoid_double(x):
    if x > 20:
        return 1.0
    elif x < -20:
        return 0.0
    return 1 / (1 + np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)


def sigmoid_prime_double(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)


def relu_double(x):
    return x if x > 0.0 else 0.01 * x


def relu(z):
    return np.vectorize(relu_double)(z)


def relu_prime_double(x):
    return 1.0 if x > 0.0 else 0.01


def relu_prime(z):
    return np.vectorize(relu_prime_double)(z)


## This is the layer superclass, it includes the basic implementations
# of the various functions that will be needed throughout the
# layer subclasses
class Layer:
    def __init__(self):
        self.params = []

        self.previous = None
        self.next = None

        self.input_data = None
        self.output_data = None

        self.input_delta = None
        self.output_delta = None

    """
    Connect allows each layer to be both a previous and a next
    
    you can then cycle through these easily
    """

    def connect(self, layer):
        self.previous = layer
        layer.next = self

    def forward(self):
        raise NotImplementedError

    """
    input is the input data for the first layer
    
    otherwise
    """

    def get_forward_input(self):
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):
        raise NotImplementedError

    def get_backward_input(self):
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta

    'Reset deltas at the end of a mini batch'

    def clear_deltas(self):
        pass

    def update_params(self, null):
        pass

    def describe(self):
        raise NotImplementedError


## The Sigmoid ActivationLayer layer subclass
# This is indicated as a separate layer to the dense layer before it
# Forward: It simply takes the output of the dense layer and applies
#    the sigmoid function to the outputs
# Backward: the derivative of the sigmoid is found
class ActivationLayer(Layer):  # noqa
    activationList = ['sigmoid', 'relu']

    def __init__(self, input_dim, activation_type='Sigmoid'):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

        self.activation_type = activation_type.lower() \
            if activation_type.lower() in self.activationList else 'sigmoid'

    def forward(self):
        data = self.get_forward_input()
        if self.activation_type == 'sigmoid':
            self.output_data = sigmoid(data)
        elif self.activation_type == 'relu':
            self.output_data = relu(data)
        else:
            raise ValueError

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        if self.activation_type == 'sigmoid':
            self.output_delta = delta * sigmoid_prime(data)
        elif self.activation_type == 'relu':
            self.output_delta = delta * relu_prime(data)
        else:
            raise ValueError

    def describe(self):
        print(self)

    def __str__(self):
        return "|-- " + self.__class__.__name__ + " - " + self.activation_type + \
               "\n\t|-- dimensions: ({})".format(self.input_dim)

    def __repr__(self):
        return str(self)


## This is the implementation of the Dense layer subclass
# This layer has weights and biases within self.parameters that
# initialise as random numbers between 0 and 1
# Forward: Takes the output of the previous layer and applies the
#   parameters to it
# Backward: Takes the backward and forward inputs.
#   Adds the backward input to delta_b, and the dot product of the
#   backward input and the forward input.
#   Output_delta is then set to be the dot product of the weights
#   and the backward input.

## I currently don't fully understand the logic behind the
# calculations for the backpropagation
class DenseLayer(Layer):  # noqa
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias]

        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias

    def backward(self):
        data = self.get_forward_input()
        delta = self.get_backward_input()
        self.delta_b += delta
        self.delta_w += np.dot(delta, data.transpose())
        self.output_delta = np.dot(self.weight.transpose(), delta)

    def update_params(self, rate):
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)

        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):
        print(self)

    def __str__(self):
        return "|-- " + self.__class__.__name__ + " - " + \
               "\n\t|-- dimensions: ({},{})".format(self.input_dim, self.output_dim)

    def __repr__(self):
        return str(self)
