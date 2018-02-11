import random
import numpy as np
from collections import deque
from keras.layers import *
from keras.models import Sequential, load_model
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.random_actions = 0

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu", input_shape=self.state_size))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters=64, kernel_size=4, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters=128, kernel_size=4, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=1))
        # model.add(Flatten())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="softmax"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Take a random action
            self.random_actions += 1
            return random.randrange(self.action_size)

        # Else: make a prediction
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        length = batch_size if batch_size <= len(self.memory) else len(self.memory) - 1
        batch = random.sample(self.memory, length)
        print("Replaying for {} experiences".format(len(batch)))
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=1)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def replay_prioritized(self, batch_size):
        batch = self.get_prioritized_batch(batch_size)
        print("Replaying for {} experiences".format(len(batch)))

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def get_prioritized_batch(self, batch_size):
        mem = np.copy(self.memory)
        mem = mem[mem[:, 2].astype(np.int).argsort()]

        length = batch_size if batch_size <= len(mem) else len(mem)-1
        return mem[: length]

    def load(self, savename):
        #self.model.load_weights(savename)
        self.model = load_model(savename)

    def save(self, savename):
        #self.model.save_weights(savename)
        self.model.save(savename)