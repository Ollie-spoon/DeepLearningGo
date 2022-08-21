from torch.nn import Module, Linear, SELU, Softmax, Dropout
from torch.utils.data import DataLoader
from torch import max
import random


def generate_batch_data(x_train, y_train, batch_size, seed=None):
    if seed is not None:
        assert (isinstance(seed, int))
    batch_seed = random.randint(0, 1000000000) if seed is None else seed
    random.seed(batch_seed)
    inputs = random.sample(x_train, batch_size)
    random.seed(batch_seed)
    labels = random.sample(y_train, batch_size)
    return inputs, labels


class MNISTModel(Module):
    """
    A dense network model with SELU activation, softmax output activation, and dropout layers after each dense layer.
    """

    def __init__(self, input_size=784, output_size=10, dropout_rate=0.2):
        """
        This initialisation is very iffy.
        The goal was to be able to input a list of sizes of dense layers and create as many or as few as needed.
        This was thwarted by the fact that PyTorch MUST HAVE parameter definitions in __init__.
        This was the only way that I could think to do this, with a dict of names of layers and layer instances

        :param size_list: A list containing the sizes of the layers between the input and output of the network. NOTE: THIS DOES NOT INCLUDE THE INPUT AND OUTPUT DIMENSIONS.
        :param input_size: The input dimensions of the network. Default=784, the default dimensions for a flattened MNIST image.
        :param output_size: The output dimensions of the network. Default=10, the number of producable options in the MNIST dataset.
        :param dropout_rate: The dropout rate of the Dropout modules after every dense layer.
        """
        super(MNISTModel, self).__init__()
        self.loss = None
        self.optimiser = None

        self.linear1 = Linear(in_features=input_size, out_features=392, bias=True)
        self.selu1 = SELU()
        self.dropout1 = Dropout(p=dropout_rate, inplace=False)
        self.linear2 = Linear(in_features=392, out_features=196, bias=True)
        self.selu2 = SELU()
        self.dropout2 = Dropout(p=dropout_rate, inplace=False)
        self.linear3 = Linear(in_features=196, out_features=output_size, bias=True)
        self.softmax3 = Softmax(dim=1)

        # self.layers = []
        # layer_dict = self.generate_layer_dict(size_list, input_size, output_size, dropout_rate)
        # for layer_name, layer_value in layer_dict.items():
        #     exec(f"self.{layer_name} = {layer_value}")
        #     self.layers.append(layer_name)

    # @staticmethod
    # def generate_layer_dict(size_list, input_size, output_size, dropout_rate):
    #     size_list.append(output_size)
    #
    #     layers = {"linear1": Linear(input_size, size_list[0])}
    #     for layer in range(1, len(size_list)):
    #         layers["selu" + str(layer)] = SELU()
    #         layers["dropout" + str(layer)] = Dropout(dropout_rate)
    #         layers["linear" + str(layer+1)] = Linear(size_list.pop(0), size_list[0])
    #     layers["softmax" + str(layer+1)] = Softmax()
    #     return layers

    def forward(self, x):

        x = self.linear1(x)
        x = self.selu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.selu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.softmax3(x)

        return x

    def compile(self, loss, optimiser):
        self.loss = loss
        self.optimiser = optimiser

    def evaluate(self, inputs_labels=None, data_loader=None):
        if inputs_labels is None and data_loader is None:
            ValueError('Either inputs_labels or data_loader must be set')
        elif inputs_labels:
            inputs = inputs_labels[0]
            labels = inputs_labels[1]
        else:
            for i, (inputs, labels) in enumerate(data_loader):
                break
            inputs = inputs.reshape(-1, 784)
        self.eval()
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        self.train()
        return loss, accuracy

    def accuracy(self, outputs, labels):
        return sum([float(outputs[i][labels[i]] == max(outputs[i])) for i in range(len(labels))]) / len(labels)

    def learn_step(self, inputs, labels):
        # zero the parameter gradients
        self.optimiser.zero_grad()

        # forward + backward + optimize
        outputs = self(inputs)
        output_loss = self.loss(outputs, labels)
        output_loss.backward()
        self.optimiser.step()
        return outputs, output_loss

    def fit(self, train_dataset, batch_size=100, epochs=5, print_interval=100):
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        for epoch in range(epochs):
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_data_loader):
                inputs = inputs.reshape(-1, 784)
                # forward + backward + optimize
                outputs, output_loss = self.learn_step(inputs, labels)

                # print statistics
                running_loss += output_loss.item()
                if i % print_interval == print_interval - 1:  # print every 100 mini-batches
                    _, accuracy = self.evaluate(inputs_labels=(inputs, labels))
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_interval:.3f} Accuracy: {accuracy:.3f}')
                    running_loss = 0.0

            print('Finished Training')

    def summary(self):
        print(self)
