import cv2
import numpy as np


def render_image(image, name=""):
    #cv2.imshow(name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imshow(name, image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def get_game_over_pix(im):
    return im[10:35, 100:225]


def save_game_over_pix(im):
    p = get_game_over_pix(im)
    pd = cv2.inRange(p, np.array([80, 80, 80]), np.array([85, 85, 85]))
    np.save("game_over_array.npy", pd)

