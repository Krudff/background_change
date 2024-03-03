import cv2
import numpy as np

# Load the video and the image
video = cv2.VideoCapture("pink.mp4")
image = cv2.imread("bg.jpeg")

while True:
    ret, frame = video.read()
    
    # Resize the frame and the image to the same size
    frame = cv2.resize(frame, (640, 480))
    image = cv2.resize(image, (640, 480))
    
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the upper and lower HSV values of the green color
    l_green = np.array([35, 100, 100])
    u_green = np.array([85, 255, 255])
    
    # Apply the mask and then use bitwise_and
    mask = cv2.inRange(hsv, l_green, u_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Subtract bitwise_and from the original green screen image
    f = frame - res
    f = np.where(f == 0, image, f)
    
    cv2.imshow("video", frame)
    cv2.imshow("mask", f)
    
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()