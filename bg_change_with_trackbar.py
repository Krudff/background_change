import cv2
import numpy as np

# opencv trackbars need callback function (but we don't need it)
def nothing():
    pass


# Create trackbars for color change
cv2.namedWindow('image')
cv2.createTrackbar('lowH','image',0,179,nothing)
cv2.createTrackbar('highH','image',179,179,nothing)

cv2.createTrackbar('lowS','image',0,255,nothing)
cv2.createTrackbar('highS','image',255,255,nothing)

cv2.createTrackbar('lowV','image',0,255,nothing)
cv2.createTrackbar('highV','image',255,255,nothing)

while True:
    # Load the video and the image
    video = cv2.VideoCapture("pink.mp4")
    image = cv2.imread("bg.jpeg")

    while True:
        ret, frame = video.read()
        
        # If the video ends, break the loop and start over
        if not ret:
            break
        
        # Resize the frame and the image to the same size
        frame = cv2.resize(frame, (640, 480))
        image = cv2.resize(image, (640, 480))
        
        # Get the new values of the trackbar
        l_h = cv2.getTrackbarPos('lowH', 'image')
        l_s = cv2.getTrackbarPos('lowS', 'image')
        l_v = cv2.getTrackbarPos('lowV', 'image')
        h_h = cv2.getTrackbarPos('highH', 'image')
        h_s = cv2.getTrackbarPos('highS', 'image')
        h_v = cv2.getTrackbarPos('highV', 'image')
        
        # Define the upper and lower BGR values of the green color
        l_green = np.array([l_h, l_s, l_v])
        u_green = np.array([h_h, h_s, h_v])
        
        # Apply the mask and then use bitwise_and
        mask = cv2.inRange(frame, l_green, u_green)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Subtract bitwise_and from the original green screen image
        f = frame - res
        f = np.where(f == 0, image, f)
        
        #cv2.imshow("video", frame)
        cv2.imshow("mask", f)
        
        if cv2.waitKey(25) == 27:
            break

    video.release()
    if cv2.waitKey(25) == 27:
        break

cv2.destroyAllWindows()