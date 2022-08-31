import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

class TransformerBlock(layers.Layer):

    def __init__(self):
        super(TransformerBlock, self).__init__()

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(input_shape=input_shape)
        self.pool1 = layers.MaxPool2D(input_shape=input_shape)





model = Sequential()
model.add(layers.Input(shape = (28,28,1)))
model.add(layers.Conv2D(32,(2,2),activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64,(2,2),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(layers.Conv2D(64*3,(2,2),activation='relu'))


model.add(layers.Dense(96, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10, activation='softmax'))

model.summary()


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
    batch_size=128,
    epochs=10)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
