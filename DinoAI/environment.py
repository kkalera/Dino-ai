"""
In this file , we will write the code that analyzes the screen
and grabs all the necessary data to train our network.

We accept up, down key's and no action as input and return the last 4 frames,
and the current score.
"""

import cv2, time
import numpy as np
import grabscreen
import pyautogui
from collections import deque


class Environment:

    def __init__(self):
        game_over_np = "game_over_array.npy"

        self.last_time = time.time()
        self.screenshot = None
        self.previous_screenshot = None
        self.available_actions = {0: "no action", 1: "arrow up", 2: "arrow down"}
        self.score = None
        self.start_time = None
        self.game_finished = False
        self.game_started = False
        self.render_game = False
        self.game_over_arr = np.load(game_over_np)
        self.buffer = deque(maxlen=4)

    def grab_screen(self):
        self.previous_screenshot = self.screenshot
        self.screenshot = grabscreen.grab_screen((10, 250, 950, 500)) #Laptop screen
        #self.screenshot = grabscreen.grab_screen((10, 350, 1910, 930)) #4k monitor
        self.buffer.appendleft(self.get_game_pixels())

        if self.render_game is True:
            render_image(self.get_game_pixels())
            """
            for i in range(len(self.buffer)):
                cv2.imshow(str(i), np.array(self.buffer[i]))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()"""

    def get_game_pixels(self):
        im = self.screenshot[0:250, 0:500]  # Laptop screen
        return im

    def get_score(self):
        self.score = int((time.time() - self.start_time) * 10)
        return self.score

    def game_over(self):
        if self.previous_screenshot is not None and self.score is not None and self.game_started is True:
            game_over_pix = self.screenshot[50:100, 300:650]
            game_over_pix_day = cv2.inRange(game_over_pix, np.array([80, 80, 80]), np.array([85, 85, 85]))
            game_over_pix_night = cv2.inRange(game_over_pix, np.array([170, 170, 170]), np.array([175, 175, 175]))

            game_over_day = np.array_equal(self.game_over_arr, game_over_pix_day)
            game_over_night = np.array_equal(self.game_over_arr, game_over_pix_night)

            if game_over_day is True or game_over_night is True:
                self.game_finished = True
                self.screenshot = None
                self.previous_screenshot = None
            else:
                self.game_finished = False

        else:
            self.game_finished = False

    def take_action(self, action):
        score = self.get_score()

        if action == self.available_actions[0]:
            pyautogui.keyUp("up")
            pyautogui.keyUp("down")
            #score += .1

        if action == self.available_actions[1]:
            pyautogui.keyDown("up")
            #score -= .1

        if action == self.available_actions[1]:
            pyautogui.keyDown("down")
            #score -= .1

        self.grab_screen()
        self.game_over()
        game_pix = np.array([pix for pix in self.buffer])
        self.last_buffer = game_pix
        return game_pix, score, self.game_finished

    def start_game(self):
        while self.screenshot is None or self.previous_screenshot is None:
            self.grab_screen()
            pyautogui.press("up")
        # Wait for the screen to change, indicating the game started
        while np.array_equal(self.screenshot, self.previous_screenshot):
            self.grab_screen()
            pyautogui.press("up")

        # Set the start time
        self.screenshot = None
        self.previous_screenshot = None
        self.start_time = time.time()
        self.game_started = True
        self.game_finished = False

    def get_ring_buffer_difference(self):
        b = np.array([p for p in self.buffer])
        return np.sum(b == self.last_buffer)


def render_image(image):
    cv2.imshow("window", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


""" Testing code """
env = Environment()
from DQNAgent import DQNAgent
max_score = 142
i=0

if __name__ == "__main__":
    agent = DQNAgent((250, 500, 3), action_size=3, memory_size=1000)
    agent.load("agent_weights.h5")
    while max_score <= 500:
        i += 1
        env.start_game()
        #env.render_game = True
        state, score, game_over = env.take_action(None)

        while not game_over:
            print(game_over)
            action = agent.act(state)
            next_state, score, game_over = env.take_action(env.available_actions[action])
            reward = score if not game_over else -10
            agent.remember(state, action, reward, next_state, game_over)
            state = next_state

        #agent.replay_prioritized(64)H

        if score > max_score:
            max_score = score
            agent.save("agent_weights.h5")
        print("test")
        print("Episode: {}, score:{}, max score: {}".format(i, score, max_score))
