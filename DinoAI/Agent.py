from keras import backend as kb
import tensorflow as tf

HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

# ----------


def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = kb.abs(err) < HUBER_LOSS_DELTA
    l2 = 0.5 * kb.square(err)
    l1 = HUBER_LOSS_DELTA * (kb.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, l2, l1) # Keras does not cover where function in tensorflow :-(

    return kb.mean(loss)


# ------------- BRAIN --------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


class Brain:
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu", input_shape=(250, 500,3)))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters=64, kernel_size=4, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters=128, kernel_size=4, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=1))
        #model.add(Flatten())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)
        model.summary()
        return model


class Agent:
    def __init__(self):
        self.brain = Brain()
        self.model = self.brain.build_model()