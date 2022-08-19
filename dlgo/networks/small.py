from tensorflow.keras.layers import Dense, Flatten, Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D(padding=3, data_format='channels_first'),
        Conv2D(48, (7, 7), activation='relu', data_format='channels_first',  input_shape=input_shape),
        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), activation='relu', data_format='channels_first'),
        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), activation='relu', data_format='channels_first'),
        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), activation='relu', data_format='channels_first'),
        Flatten(),
        Dense(512, activation='relu')
    ]
