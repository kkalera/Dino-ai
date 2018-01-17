from screengrabber2 import ScreenGrabber
from multiprocessing import Manager
import time
import numpy as np
import helper


class Env:
    def __init__(self, render=True, resize=True, resize_size=(320,160)):
        self.mgr = Manager()
        self.sc = ScreenGrabber(bbox=(10, 450, 1910, 850), render=render, resize=resize, resize_size= resize_size)

    def start_game(self):
        # Start the screen grabber
        self.sc.process.start()

    def get_screen(self):
        screen = np.load('screen.npy')
        print(screen.shape)
        helper.render_image(screen[0])


def p(x):
    while True:
        print(len(x.get_screens()))


if __name__ == "__main__":

    env = Env()
    env.start_game()

    time.sleep(3)
    #env.sc.stop()
    while True:
        env.get_screen()




