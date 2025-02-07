import cv2
import numpy as np

while True:
    im_out = np.zeros((200, 200, 3), np.uint8)
    cv2.imshow("win", im_out)
    key = cv2.waitKeyEx(1)
    if key >= 0:
        print(hex(key))
