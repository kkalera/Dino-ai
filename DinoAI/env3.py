import time
import helper
from DQNAgent import DQNAgent
import threading
from collections import deque
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from selenium import webdriver  
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains


class DinoEnvironment:

    CHROME_PATH = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
    CHROMEDRIVER_PATH = "C:\\Program Files (x86)\\ChromeDriver\\chromedriver.exe"
    WINDOW_SIZE = "800,600"

    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    chrome_options.binary_location = CHROME_PATH

    def __init__(self, pause_between_actions=False, render=False):
        self.driver = self.get_driver("chrome://dino")
        self.pause_between_actions = pause_between_actions
        self.action_space = {0: "no action", 1: "arrow up", 2: "arrow down"}
        self.render = render
        self.start_time = time.time()
        self.done = False
        self.checker_running=False

    def get_driver(self, url):

        d = webdriver.Chrome(
            executable_path=self.CHROMEDRIVER_PATH,
            chrome_options=self.chrome_options
        )
        d.get(url)
        return d

    def get_game_activated(self, d):
        c = "return Runner.instance_.activated"
        return d.execute_script(c)

    def get_game_paused(self, d):
        c = "return Runner.instance_.paused"
        return d.execute_script(c)

    def set_game_paused(self, d):
        c = "Runner.instance_.stop()"
        return d.execute_script(c)

    def set_game_playing(self, d):
        c = "Runner.instance_.play()"
        return d.execute_script(c)

    def get_game_over(self):
        command = "return Runner.instance_.crashed"
        self.done = self.driver.execute_script(command)
        return self.done

    def key_up_press(self, d):
        canvas = d.find_element_by_class_name("runner-canvas")
        ac = ActionChains(d)
        ac.key_down(Keys.ARROW_UP, canvas)
        ac.key_up(Keys.ARROW_UP, canvas)
        ac.perform()

    def key_up_release(self, d):
        canvas = d.find_element_by_class_name("runner-canvas")
        ac = ActionChains(d)
        ac.key_up(Keys.ARROW_UP, canvas)
        ac.perform()

    def key_down_press(self, d):
        canvas = d.find_element_by_class_name("runner-canvas")
        ac = ActionChains(d)
        ac.key_down(Keys.ARROW_DOWN, canvas)
        ac.key_up(Keys.ARROW_DOWN, canvas)
        ac.perform()

    def key_down_release(self, d):
        canvas = d.find_element_by_class_name("runner-canvas")
        ac = ActionChains(d)
        ac.key_up(Keys.ARROW_DOWN, canvas)
        ac.perform()

    def get_distance_ran(self, d):
        c = "return Math.ceil(Runner.instance_.distanceRan)"
        return d.execute_script(c)

    def reset(self):
        self.checker_running = False

        # Start the game if not yet started
        while not self.get_game_activated(self.driver):
            self.key_up_press(self.driver)
            self.key_up_release(self.driver)

        # Reset the game
        c = "Runner.instance_.restart()"
        self.driver.execute_script(c)
        self.start_time = time.time()

        # Get 4 screenshots
        s = []
        while len(s) < 4:
            s.append(self.get_screen(self.driver))

        s = self.get_normalized_input(s)

        return s, self.get_distance_ran(self.driver), self.get_game_over()

    def get_screen(self, d):
        element = d.find_element_by_class_name("runner-canvas")
        img = d.execute_script("return arguments[0].toDataURL('img/png').substring(21)", element)
        img = base64.b64decode(img)
        img = Image.open(BytesIO(img)).convert('L')  # Convert it to grayscale
        # img = Image.open(BytesIO(img))  # Without converting
        img = img.crop((0, 0, 300, 150))
        img = np.asarray(img)

        # Render the image if enabled:
        if self.render:
            helper.render_image(img)

        return img

    def get_normalized_input(self, s):
        a = np.array(s)
        a_norm = a * (255 / a.max()) / 255
        return a_norm

    def act(self, action):
        if self.get_game_over() is False:

            # Do the action
            """if action == self.action_space[0]:
                self.key_up_release(self.driver)
                self.key_down_release(self.driver)"""

            if action == self.action_space[1]:
                self.key_up_press(self.driver)

            if action == self.action_space[2]:
                self.key_down_press(self.driver)

        # Get the screens:
        s = []
        while len(s) < 4:
            s.append(self.get_screen(self.driver))

        # Get our images normalized
        s = self.get_normalized_input(s)

        # Return the values
        return s, self.get_distance_ran(self.driver), self.get_game_over(self.driver)


if __name__ == '__main__':
    env = DinoEnvironment()
    agent = DQNAgent((150, 300), action_size=3, memory_size=5000)
    max_score = 0

    for i in range(100):
        state, score, done = env.reset()
        print("state shape: {}".format(state.shape))
        a = 0

        while env.get_game_over() is False:
            action = agent.act(state)
            next_state, score, done = env.act(env.action_space[action])

            reward = score if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            a += 1

        print()
        print("Game Over")

        if score > max_score:
            max_score = score
            agent.save("agent.h5")

        aps = a / (time.time() - env.start_time)
        print("Epoch: {}, score:{}, max score: {}, act/s: {:.2f}, {}/{} actions"
              " were random".format(i, score, max_score, aps, agent.random_actions, agent.actions_taken))

        # Reset the actions
        agent.actions_taken, agent.random_actions = 0, 0

        # Replay experiences
        agent.replay(2024)