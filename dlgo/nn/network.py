import random
import numpy as np
import pickle
import os


class MSE:

    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions, labels):
        diff = predictions - labels
        return 0.5 * sum(diff * diff)[0]

    @staticmethod
    def loss_derivative(predictions, labels):
        return predictions - labels


class SequentialNetwork:

    ## create a network with no layers and a loss function
    def __init__(self, loss=None):
        print("Initialising Network...")
        self.layers = []
        if loss is None:
            self.loss = MSE()  # Please implement

    ## add a layer to self.layers then connect it to the
    # previous layer
    def add(self, layer):
        self.layers.append(layer)
        ## layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    ## To train the network:
    #   iterate through epochs
    #       separate the training data into sections of size mini_batch_size
    #       iterate through mini_batches and train data
    #       if there is test data then evaluate the test_data
    #       either way, output results
    def train(self, training_data, epochs, mini_batch_size,
              learning_rate, test_data=None):
        n = len(training_data)
        n_test = len(test_data)
        ## self.save(extension='initial')
        if test_data:
            print("Initial: {0} / {1}"
                  .format(self.evaluate(test_data), n_test))
        for epoch in range(epochs):
            print("epoch: " + str(epoch))
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for
                k in range(0, n, mini_batch_size)  # range(start, end, increment)
            ]
            i = 0.0
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, learning_rate)
                i += 1.0
                if test_data and i/5.0 == np.floor(i/5.0):
                    print("Apres-Mini: {0} / {1}"
                          .format(self.evaluate(test_data), n_test))
            ## self.save(extension=epoch)
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}"
                      .format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def train_batch(self, mini_batch, learning_rate):
        self.forward_backward(mini_batch)
        self.update(len(mini_batch), learning_rate)

    ## Learning rate is normalised by the batch size
    def update(self, mini_batch_size, learning_rate):
        learning_rate = learning_rate / float(mini_batch_size)
        for layer in self.layers:
            layer.update_params(learning_rate)
        for layer in self.layers:
            layer.clear_deltas()

    def forward_backward(self, mini_batch):
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:
                layer.forward()
            self.layers[-1].input_delta = \
                self.loss.loss_derivative(self.layers[-1].output_data, y)
            for layer in reversed(self.layers):
                layer.backward()

    def single_forward(self, x):
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data):
        test_results = [(
            np.argmax(self.single_forward(x)),
            np.argmax(y)
        ) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def save(self, extension=''):
        if type(extension) != str:
            extension = str(extension)
        if extension != '':
            extension = '_' + extension
        address = "save_data/SeqNNmnist" + extension + ".pkl"
        with open(address, "wb") as file:
            pickle.dump({'net': self}, file)
        print(self.__class__.__name__ + " saved to " + address + ".")

    def __str__(self):
        out = self.__class__.__name__ + '\n'
        for layer in self.layers:
            out += ' ' + layer.__repr__() + '\n'
        return out

    def __repr__(self):
        return str(self)
