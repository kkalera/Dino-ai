from selenium.webdriver import Chrome, ActionChains, ChromeOptions
from selenium.webdriver.common.keys import Keys
import base64
from PIL import Image
from io import BytesIO
from DinoAI import helper
import numpy as np
from DinoAI.DQNAgent import DQNAgent


class DinoEnvironment:
    def __init__(self, window_size="800,600", render=False):
        self.window_size = window_size
        self.render = render
        self.browser = self.init_browser()

    def init_browser(self):
        options = ChromeOptions()
        options.add_argument("--window-size=%s" % self.window_size)
        browser = Chrome(
            executable_path="c:/program files (x86)/chromedriver/chromedriver.exe",
            options=options)
        browser.get("chrome://dino")
        return browser

    def key_toggle_up(self):
        ac = ActionChains(self.browser)
        ac.send_keys(Keys.ARROW_UP)
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

        # Render the image if enabled:
        if self.render is True:
            helper.render_image(img)

        return img


if __name__ == "__main__":
    env = DinoEnvironment(render=False)
    agent = DQNAgent((4, 75, 150), action_size=2, memory_size=3000)

    while env.game_over() is False:
        env.key_toggle_up()
        print(env.get_score())

