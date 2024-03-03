import cv2
import numpy as np

video = cv2.VideoCapture("green.mp4")
image = cv2.imread("bg.jpeg")

while True:

    ret, frame = video.read()

    frame = cv2.resize(frame, (640, 480))
    image = cv2.resize(image, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    u_green = np.array([104, 153, 70])#upper green (upper color boundary)
    l_green = np.array([30, 30, 0])#lower green (lower color boundary)

    #below creates a mask where green colors (within boundary) are set to white, and the rest black
    mask = cv2.inRange(frame, l_green, u_green)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    #remove background
    bg_removed = frame - res

    #if black pixel: replace with background image; if not: retain
    bg_change = np.where(bg_removed==0, image, bg_removed)

    cv2.imshow("video", frame)
    #cv2.imshow("image",image)
    #cv2.imshow("mask", mask)
    #cv2.imshow("res", res)
    #cv2.imshow("background removed", bg_removed)
    cv2.imshow("background change", bg_change)

    


    cv2.waitKey(25) == 27

    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()
