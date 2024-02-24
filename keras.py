import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from keras_core import Sequential
from keras_core.layers import \
    Dense, Flatten, Conv2D, Dropout, \
    BatchNormalization, ReLU
from keras_core.optimizers import Adam
from keras_core.losses import SparseCategoricalCrossentropy
from keras_core.metrics import SparseCategoricalAccuracy
from keras_core.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 127.5 - 1.0
x_test = np.expand_dims(x_test, axis=-1) / 127.5 - 1.0

model = Sequential([
    BatchNormalization(),
    Conv2D(kernel_size=(5, 5), filters=32, strides=(2, 2), padding="same"),
    Dropout(rate=0.3),
    ReLU(),
    Conv2D(kernel_size=(3, 3), filters=64, strides=(2, 2), padding="same"),
    Dropout(rate=0.3),
    ReLU(),
    Conv2D(kernel_size=(3, 3), filters=128, strides=(2, 2), padding="same"),
    Dropout(rate=0.3),
    ReLU(),
    Conv2D(kernel_size=(3, 3), filters=256, strides=(2, 2), padding="same"),
    Dropout(rate=0.3),
    ReLU(),
    Flatten(),
    Dense(10, activation="softmax")
])
model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=[SparseCategoricalAccuracy()]
)
model.build((None, 28, 28, 1))

model.fit(
    epochs=10, batch_size=64,
    x=x_train, y=y_train,
    validation_data=(x_test, y_test)
)
