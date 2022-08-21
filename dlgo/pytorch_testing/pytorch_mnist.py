"""
NOTE: This version of PyTorch mnist uses the general formatting and structure of dlgo/keras_testing/keras_mnist.py
    rather than the 'PyTorch way of doing things'. This is also an amalgamation of different scraps of what I have
    managed to rip from various places on the internet combined with my own approach to making it work.
    It is therefore not following industry best practices and is likely highly inefficient.

Important differences to note between keras and pytorch.
    Generators become DataLoaders
    No Compiling step
    No apparent fit function
    Learning rate must be specified
    Dropout has an inplace setting set by default to False. This means that the dropout operation
        is applied directly on the output of the previous layer rather than on a copy of the output.
        Don't use until in early stages.
    In this example at least Generators/DataLoaders seem to be a mandatory requirement. (not a bad thing)

Main steps:
1. Import data/create data_loaders.
2. Create a model (I hate the way that this is unautomisable)
3.

TODO: Figure out how to flatten the train_dataset and test_dataset upon import.
TODO: Combine the test_data_loader into the MNISTModel in mnist_model.py

"""

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from dlgo.pytorch_testing.mnist_model import MNISTModel


# <1>

train_dataset = MNIST(root='./data',
                      train=True,
                      transform=ToTensor(),
                      download=True)
test_dataset = MNIST(root='./data',
                     train=False,
                     transform=ToTensor())

# <2>

model = MNISTModel()

# <3>

loss = CrossEntropyLoss()
optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)

model.compile(loss=loss, optimiser=optimiser)

# <4>
model.fit(train_dataset,
          batch_size=120,
          epochs=10)

score = model.evaluate_test(test_data=test_dataset)
print('Test loss:', score[0].item())
print(f'Test accuracy: ', score[1])
