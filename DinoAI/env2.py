import grabscreen
import cv2
import helper
from collections import deque
from screengrabber import ScreenGrabber
import numpy as np
import pyautogui
import time
from DQNAgent import DQNAgent
from skimage import transform


class Environment:
    def __init__(self, render=False):
        self.pix = None
        self.prev_pix = None
        self.available_actions = {0: "no action", 1: "arrow up", 2: "arrow down"}
        self.score = 0
        self.finished = False
        self.started = False
        self.mem = deque(maxlen=4)
        self.start_time = None
        self.game_over_arr = np.load("game_over_array.npy")

        if render:
            s = ScreenGrabber((10, 450, 1910, 850), render=True, resize=True)
            s.start()

    def get_game_pixels(self):
        im = grabscreen.grab_screen((10, 450, 1910, 850))
        im = cv2.resize(im, (320, 160))
        self.mem.append(im)
        self.prev_pix = self.pix
        self.pix = im
        return im

    def start_game(self):
        # Get the first game pixels
        while self.pix is None or self.prev_pix is None:
            self.get_game_pixels()

        # Wait until the frames are different, indicating the game has started
        while np.array_equal(self.pix, self.prev_pix):
            self.get_game_pixels()
            pyautogui.press("up")

        # Reset all values
        self.pix = None
        self.prev_pix = None
        self.start_time = time.time()
        self.started = True
        self.finished = False

        # Fill the memory with fresh images
        for i in range(4):
            self.get_game_pixels()

        return self.get_pix_arr(self.mem), self.score, self.finished

    def act(self, action):
        # Release all keys at "no action"
        if action == self.available_actions[0]:
            pyautogui.keyUp("up")
            pyautogui.keyUp("down")

        # Press down up key at "arrow up"
        if action == self.available_actions[1]:
            pyautogui.keyDown("up")

        # Press down down key at "arrow down"
        if action == self.available_actions[2]:
            pyautogui.keyDown("down")

        # Grab the new pixels
        for i in range(4):
            self.get_game_pixels()

        # Check if the game is over
        self.check_game_state()

        # Calculate the score
        self.update_score()

        return self.get_pix_arr(self.mem), self.score, self.finished

    def get_pix_arr(self, arr):
        # Get our array of images
        a = np.stack(arr)
        a_gray = np.sum(a/3, axis=3, keepdims=True)
        a_norm = a_gray * (255 / a_gray.max()) / 255
        a_norm = a_norm.reshape(1,a_norm.shape[2], a_norm.shape[1], a_norm.shape[0])
        return a_norm

    def check_game_state(self):
        if self.started:
            pix = helper.get_game_over_pix(self.pix)
            game_over_pix_day = cv2.inRange(pix, np.array([80, 80, 80]), np.array([85, 85, 85]))
            game_over_pix_night = cv2.inRange(pix, np.array([170, 170, 170]), np.array([175, 175, 175]))

            #game_over_day = np.array_equal(self.game_over_arr, game_over_pix_day)
            #game_over_night = np.array_equal(self.game_over_arr, game_over_pix_night)

            gd = np.count_nonzero(self.game_over_arr == game_over_pix_day)
            gn = np.count_nonzero(self.game_over_arr == game_over_pix_night)
            #print(gd, gn)
            if(gd > 2900) or (gn > 2900) or np.array_equal(self.pix, self.prev_pix):
                self.finished = True
                #self.started = False

            #if game_over_day or game_over_night:
             #   self.finished = True
             #   self.started = False

        else:
            self.finished = False

    def update_score(self):
        self.score = int((time.time() - self.start_time) * 10)



if __name__ == "__main__":
    # Create the environment
    env = Environment(render=False)
    agent = DQNAgent((320, 160, 4), action_size=3, memory_size=5000)
    agent.load("agent.h5")
    max_score = 146

    for i in range(1000):
        # Get the initial values
        state, score, done = env.start_game()
        a = 0
        # Loop until game over
        while not done:
            action = agent.act(state)
            next_state, score, done = env.act(env.available_actions[action])
            reward = score if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            a += 1
            helper.render_image(state[0])

        print("Game over")
        if score > max_score:
            max_score = score
            agent.save("agent.h5")

        aps = a/(time.time() - env.start_time)
        print("Episode: {}, score:{}, max score: {}, actions/sec: {}".format(i, score, max_score, aps))

        agent.replay(2024)
