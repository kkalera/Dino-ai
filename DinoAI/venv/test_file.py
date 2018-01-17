import numpy as np
import cv2
import grabscreen
"""
a = np.array([["value1", "value2", 3, "value4", "value5"],
             ["value6", "value7", -10, "value8", "value9"],
             ["value10", "value11", 31, "value12", "value13"],
             ["value14", "value15", 5, "value16", "value17"],
             ["value18", "value19", 3, "value20", "value21"]])

print("Default")
print(a)

a = a[a[:, 2].astype(np.int).argsort()]

print()
print("Sorted:")
print(a)"""
# 3,840x2,160
def grab():
    im = grabscreen.grab_screen((10, 350, 1910, 930))
    render_image(im)

def render_image(image):
    #cv2.imshow("window", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imshow("1", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


while True:
    grab()
