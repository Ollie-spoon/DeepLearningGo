"""
Important differences to note between keras and pytorch.
    Generators become DataLoaders
    No Compiling step
    No apparent fit function
    Learning rate must be specified
    Dropout has an inplace setting set by default to False. This means that the dropout operation
        is applied directly on the output of the previous layer rather than on a copy of the output.
        Don't use until in early stages.
    In this example at least Generators/DataLoaders seem to be a mandatory requirement. (not a bad thing)


TODO: Figure out how to flatten the train_dataset and test_dataset upon import.

"""

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from dlgo.pytorch_testing.mnist_model import MNISTModel


print('----imports complete')

# <1>

train_dataset = MNIST(root='./data',
                      train=True,
                      transform=ToTensor(),
                      download=True)
test_dataset = MNIST(root='./data',
                     train=False,
                     transform=ToTensor())
test_data_loader = DataLoader(dataset=test_dataset, shuffle=False)

# <2>


# def __repr__(self):
#     return str(self.layers)
model = MNISTModel()

model.summary()

# <3>

loss = CrossEntropyLoss()
optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)

model.compile(loss=loss, optimiser=optimiser)

# <4>
model.fit(train_dataset,
          batch_size=120,
          epochs=2)

score = model.evaluate(data_loader=test_data_loader)
print('Test loss:', score[0].item())
print(f'Test accuracy: {score[1]:f5}')
