from collections import deque
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import numpy as np


class TestClass:
    def __init__(self):
        self.mem = deque(maxlen=4)
        self.process = Process(target=self.run)

    def run(self):
        while True:
            self.mem.append(np.array([0, 1, 2, 3, 4]))


def print_values(x):
    while True:
        print(x)


test = TestClass()
process = Process(target=print_values(test.mem))

test.process.start()
process.start()
