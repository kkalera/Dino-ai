from selenium.webdriver import Chrome, ActionChains, ChromeOptions
from selenium.webdriver.common.keys import Keys
import base64
from PIL import Image
from io import BytesIO
from DinoAI import helper
import numpy as np
from DinoAI.DQNAgent import DQNAgent
import time
import random
import cv2


class DinoEnvironment:
    def __init__(self, window_size="800,600", render=False):
        self.window_size = window_size
        self.render = render
        self.browser = self.init_browser()
        self.actions = {0: "Do nothing", 1: "Jump"}
        self.start_time = time.time()

    def init_browser(self):
        options = ChromeOptions()
        options.add_argument("--window-size=%s" % self.window_size)
        options.add_argument("--mute-audio")
        browser = Chrome(
            executable_path="c:/program files (x86)/chromedriver/chromedriver.exe",
            options=options)
        browser.get("chrome://dino")
        return browser

    def key_toggle_up(self):
        ac = ActionChains(self.browser)
        ac.send_keys(Keys.ARROW_UP)
        ac.perform()

    def key_toggle_down(self):
        ac = ActionChains(self.browser)
        ac.send_keys(Keys.ARROW_DOWN)
        ac.perform()

    def game_over(self):
        return self.browser.execute_script("return Runner.instance_.crashed")

    def game_activated(self):
        return self.browser.execute_script("return Runner.instance_.activated")

    def get_score(self):
        # Executing this script in the browser, returns an
        # array of the 5 digits that display the score.
        score_digits = self.browser.execute_script("return Runner.instance_.distanceMeter.digits")

        # Here we join all the returned digits together into a single string
        score_digits = "".join(score_digits)

        # If the game is not running yet, the score digits will be "" to prevent an error we return 0
        if score_digits == "":
            return 0

        # Finally we convert the digits to integers and return the result
        return int(score_digits)

    def get_screen(self):
        d = self.browser
        element = d.find_element_by_class_name("runner-canvas")
        img = d.execute_script("return arguments[0].toDataURL('img/png').substring(21)", element)
        img = base64.b64decode(img)
        img = Image.open(BytesIO(img)).convert('L')  # Convert it to grayscale
        # img = Image.open(BytesIO(img))  # Without converting
        img = img.crop((0, 0, 300, 150))
        img = np.asarray(img)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # Resize it from 300x150 to 150x75

        # Render the image if enabled:
        if self.render is True:
            helper.render_image(img)

        return img

    def get_refresh_rate(self):
        return self.browser.execute_script("return Runner.instance_.msPerFrame")

    def act(self, action):
        # Check if the action is in the action space
        assert action in self.actions.keys()

        # Perform the action
        if action == 1:
            self.key_toggle_up()

        rr = self.get_refresh_rate()
        frames = []
        for i in range(4):
            frames.append(self.get_screen())
            time.sleep(rr/1000)

        return np.expand_dims(np.array(frames), axis=0), self.get_score(), self.game_over()

    def reset(self):
        self.browser.execute_script("Runner.instance_.restart()")
        self.start_time = time.time()
        return self.act(0)


if __name__ == "__main__":
    env = DinoEnvironment(render=False)
    agent = DQNAgent((4, 75, 150), action_size=2, memory_size=3000)
    epochs = 2
    high_score = 0

    for i in range(epochs):
        actions_taken = 0
        state, score, done = env.reset()  # Reset the environment
        agent.random_actions = 0

        while env.game_over() is False:
            action = agent.act(state)
            next_state, score, done = env.act(action)
            reward = score if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            actions_taken += 1

        print()
        print("Game over")

        if score > high_score:
            high_score = score
            agent.save("agent.h5")

        aps = actions_taken / (time.time() - env.start_time)
        print("Epoch: {}/{}, score: {}, high_score: {}, act/s: {:.2f}, {}/{} actions"
              " were random".format(i+1, epochs, score, high_score, aps,
                                    agent.random_actions, actions_taken))

        agent.replay(2024)

    print()
    print("Done training")
    env.browser.close()