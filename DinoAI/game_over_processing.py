import grabscreen
import numpy as np
import cv2

def render_image(image):
    #cv2.imshow("window", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imshow("window", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

screenshot = grabscreen.grab_screen((10, 250, 950, 500))
print(screenshot.shape)
game_over_image = screenshot[50:100, 300:650]
game_over_image = cv2.inRange(game_over_image, np.array([170, 170, 170]), np.array([175, 175, 175]))
print(game_over_image.shape)
#np.save("game_over_array.npy", game_over_image)
while True:
    screenshot = grabscreen.grab_screen((10, 250, 950, 500))
    print(screenshot.shape)
    game_over_image = screenshot[50:100, 300:650]
    game_over_image = cv2.inRange(game_over_image, np.array([170, 170, 170]), np.array([175, 175, 175]))
    print(game_over_image.shape)
    render_image(game_over_image)

