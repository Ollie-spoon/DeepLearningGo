import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling1D, Dropout
from transformer import TransformerBlock, TokenAndPositionEmbedding

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

inputs = Input(shape=(784,))
embedding_layer = TokenAndPositionEmbedding(200, 2000, 32)
transformer_block = TransformerBlock(32, 2, 32)

model = Sequential()
model.add(embedding_layer(inputs))
model.add(transformer_block)
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.1))
model.add(Dense(96, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='sigmoid'))

# model.add(Dense(392, activation='selu', input_shape=(784,)))
# model.add(Dense(196, activation='selu'))
# model.add(Dense(10, activation='sigmoid'))
# model.summary()



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
    batch_size=128,
    epochs=10)
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])