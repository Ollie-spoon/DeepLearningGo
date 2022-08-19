# tag::train_generator_imports[]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder

from dlgo.networks import small
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint  # <1>

# <1> With model checkpoints we can store progress for time-consuming experiments <3
# end::train_generator_imports[]

# tag::train_generator_generator[]
if __name__ == '__main__':
    go_board_size = 19
    num_classes = go_board_size ** 2
    num_games = 1000

    encoder = OnePlaneEncoder((go_board_size, go_board_size))  # <1>

    processor = GoDataProcessor(encoder=encoder.name())  # <2>

    generator = processor.load_go_data('train', num_games, use_generator=True)  # <3>
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    # <1> First we create an encoder of board size.
    # <2> Then we initialize a Go Data processor with it.
    # <3> From the processor we create two data generators, for training and testing.
    # end::train_generator_generator[]

    # tag::train_generator_model[]
    input_shape = (encoder.num_planes, go_board_size, go_board_size)
    model = Sequential()
    for layer in small.layers(input_shape):
        model.add(layer)
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # end::train_generator_model[]

    # tag::train_generator_fit[]
    epochs = 20
    batch_size = 128
    model.fit(generator.generate(batch_size, num_classes),  # <1>
              epochs=epochs,
              steps_per_epoch=generator.get_num_samples() / batch_size,  # <2>
              validation_data=test_generator.generate(batch_size, num_classes),  # <3>
              validation_steps=test_generator.get_num_samples() / batch_size,  # <4>
              callbacks=[ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.h5')])  # <5>

    model.evaluate(test_generator.generate(batch_size, num_classes),
                   steps=test_generator.get_num_samples() / batch_size)  # <6>
# <1> We specify a training data generator for our batch size...
# <2> ... and how many training steps per epoch we execute.
# <3> An additional generat or is used for validation...
# <4> ... which also needs a number of steps.
# <5> After each epoch we persist a checkpoint of the model.
# <6> For evaluation we also speficy a generator and the number of steps.
# end::train_generator_fit[]
