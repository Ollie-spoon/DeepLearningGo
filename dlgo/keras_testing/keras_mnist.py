"""
Using Keras is very simple and takes just 4 steps
1. Preprocessing of data to fit the desired network.
        This includes the creation of test data
2. Creating an architecture
3. Compiling the model with a choice of metrics
4. Fitting the model to the data

Points to note:

Activation functions:
SELU typically works better than sigmoid/softmax as an activation function
        SELU: alpha*(e^x - 1) below 0
The final layer of the network should usually have a softmax activation 
        as this places the final value between 0-1 and 'normalises' the 
        output values. - Sum(output) = 1.0

Loss functions:
When compiling, Mean_Squared_Error (MSE) works very well for regression 
        problems; problems with continuous range outputs. (between 0 and 
        1 but not specifically either one)
For binary output examples, categorical_crossentropy (CCE) works much better 
        as the loss is only measured on the single value that should have 
        been selected.
More details for this are on p. 139, section 6.5.2 - Makes a lot of sense

Optimizer:
Decay - (of learning rate) - Both decay and momentum on p.171(200)
Momentum - Momentum adds a portion of the last update to the next update.
        This has the effect of little to no effect, unless the target has
        been overshot, at which point the two vectors will cancel out
        somewhat and will hopefully be closer to the minimum.
SGD is typically slower but more accurate than ADAM, and learning for the 
        SGD:MSE combination slows down over time. However, categorical
        crossentropy pairs brilliantly with SGD and learning doesn't slow
        down as you approacch higher prediction values.
Adagrad, Adadelta both use adaptive learning rates meaning that the learning
        rates for different neurons aren't necesarrily the same. For example
        neurons corresponding to frequently observed behaviours will learn
        less and less over time, and more niche behaviours that occur
        infrequently in training data have a higher learning rate.
Adagrad. Doesn't have a global learning rate, but instead has a different
        learning rate for each parameter. The magnitude of updates for each
        parameter gets slowly smaller over time. This works really well when
        you have a large dataset with lots of different niche behaviours
        that occur infrequently.
Adadelta. This works similarly to Adagrad except a 'momentum' style functions
        is used to decrease the learning rate per perameter over time.


Layers:
Dense. This is the standard layer of perceptrons, with a choice of actiation 
        function.
Dropout. This removes a certain number of nodes in the previous step during 
        train to ensure that the network works well on a a more general set 
        of cases and doesn't rely too heavily on a single neuron.
Pooling. Like a convolutional layer but instead of applying a fliter it takes 
        one of [Max, Min, Mean] from the values within the filter. Each of 
        these options has it's pros and cons; I don't understand why yet but 
        for the go_cnn file after 100 epochs Max: 0.83%, Mean: 0.63%.
        Allows the size of a section of data to be reduced massively while 
        not sacraficing too much quality.

Tuning Architectures and Hyperparameters guide:
When thinking about convolution consider what changes will have to an image
        being convolved.
Convolution (followed by at least 2 dense layers including the output) works
        significantly better than just dense for images. (Now probably
        replaced by transformers given enough training data)
For Conv layers kernel size can make a large difference to performance,
        typically don't go above 7.
Pooling layers can improve efficiencies, try max, average, without. Keep the
        pooling size within reason, 3 should definitely be large enough.
Dropout layers are fantastic, but don't use too many of them and keep the
        dropout rate at a reasonable level.
Again, last layer: softmax for probability distribution. This works very well
        for categorical cross-entropy loss.
ReLU, elu, selu, PReLU, and LeakyReLU all work very well, there are also
        others that aren't variants of ReLU but ReLU works brilliantly as a
        benchmark.
In theory you want every mini batch to contain information on every possible
        behaviour that you are trying to learn. However, in practice this
        can't always work out perfectly but take this into consideration when
        selecting mini batch size.
Carefully consider which optimiser you should use, they all have pros and cons
        and all are the best optimiser for a certain type of problem.
        -- keras.io/api/optimizers/
Number of epochs is important; generally go on the large size. This is only
        ever an issue when dealing with over-fitting which can be a major
        problem. To counter this issue, we have the model checkpointing.
One extra super important thing to consider is that random numbers are often
        not a good idea when initialising your weights/parameters. There is an
        entire library of different initialisers to chose from; These can be
        specified on a layer by layer basis. (The keras defaults are typically
        very good but it's worth considering to give yourself a head start in
        training)
        -- https://keras.io/api/layers/initializers/

General guidelines/Helpful pointers:
Validation loss over time is very important. When using checkpointing use the
        model from the epoch that has both high training and validation accuracy.
        Watch out for overfitting here, as the the validation loss will start to
        increase when the model starts to overfit.
With a low training error but a large validation error, you likely do not have
        enough data (or maybe your mini-batch size is too small).
AWS can be used to perform training, this is all well documented in Appendix D
        p. 326(355)
When comparing runs, don’t stop a run that looks worse than a previous run too
        early. Some learning processes are slower than others—and might
        eventually catch up or even outperform other models.
"""

from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

# <1>

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# <2>

model = Sequential()
model.add(Dense(392, activation='selu', input_shape=(784,)))
model.add(Dropout(0.1))
model.add(Dense(196, activation='selu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))
model.summary()

# <3>

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# <4>

model.fit(x_train, y_train,
    batch_size=128,
    epochs=20)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
