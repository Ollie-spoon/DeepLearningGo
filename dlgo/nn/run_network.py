from dlgo.nn import load_mnist
from dlgo.nn import network
from dlgo.nn.layers import DenseLayer, ActivationLayer
import pickle
import os
import time
os.system('color')

training_data, test_data = load_mnist.load_data()
activation_functions = ['Sigmoid', 'ReLu']
load = False

if load:
    address = "save_data/SeqNNmnist_0.pkl"
    SeqNNmnist = pickle.load(address)
    print("SeqNNmnist: " + str(SeqNNmnist))
    net = SeqNNmnist['net']
    print("net: " + str(net))
else:
    start = time.time()
    net = network.SequentialNetwork()

    act = activation_functions[0]

    net.add(DenseLayer(784, 392))
    net.add(ActivationLayer(392, activation_type=act))
    net.add(DenseLayer(392, 196))
    net.add(ActivationLayer(196, activation_type=act))
    net.add(DenseLayer(196, 10))
    net.add(ActivationLayer(10, activation_type='Sigmoid'))

    print("\n\033[96mNetwork initialised.\033[0m\n")

    net.train(training_data, epochs=10, mini_batch_size=20,
          learning_rate=1.0, test_data=test_data)
    print('time taken: ' + str(time.time() - start) + 's')
