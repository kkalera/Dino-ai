import multiprocessing
import cv2
import grabscreen
import helper
from collections import deque
import numpy as np


class ScreenGrabber():
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, bbox, render=False, resize=False, resize_size=(320, 160)):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.bbox = bbox
        self.screen_mem = deque(maxlen=4)
        self.render = render
        self.done = True
        self.resize = resize
        self.resize_size = resize_size
        self.exit = multiprocessing.Event()
        self.process = multiprocessing.Process(target=self.run)

    def run(self):
        """ Method that runs forever """
        while True:
            # If the exit is set, quit the process
            if self.exit.is_set():
                cv2.destroyAllWindows()
                break

            # Do something
            if self.done:

                # Prevent another grab from happening until this one is finished
                self.done = False

                # Grab the pixels
                im = grabscreen.grab_screen(region=self.bbox)

                # Resize if allowed
                if self.resize:
                    im = cv2.resize(im, self.resize_size)

                # Render if allowed
                if self.render is True:
                    helper.render_image(im)

                # Add the image to the memory
                self.screen_mem.append(im)
                np.save("screen.npy", np.array(self.screen_mem))

                # Allow the next grab to happen
                self.done = True

    def stop(self):
        # Set the exit Event
        self.exit.set()

    def get_screens(self):
        # function that return the 4 last captured screenshots
        return self.screen_mem
