import cv2
import mediapipe as mp
import numpy as np

# Initialize the selfie segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 1)

# Read the background image
bg_image = cv2.imread('sajang.png')

######
# Open the video file
cap = cv2.VideoCapture('vid2.mp4')

# Open the webcam
#cap = cv2.VideoCapture(0)
#####


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = mp_selfie_segmentation.process(frame_rgb)

    # Create an empty mask with the same shape as the frame
    mask = np.zeros(frame.shape, dtype=np.uint8)

    # Get the condition where segmentation mask is 1 (i.e., person is present)
    condition = results.segmentation_mask > 0.1

    # Apply the condition to the frame
    mask[condition] = frame[condition]

    # Resize the background image to match the frame size
    bg_image_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))

    ##########
    # Apply the condition to the background image
    #bg_image_resized[condition] = frame[condition]

    # Create a blurred version of the condition #(25,25) is the kernel size (less will make it sharper)
    blurred_condition = cv2.GaussianBlur(condition.astype(np.float32), (25, 25), 0)

    # Apply the blurred condition to the background image
    bg_image_resized = (bg_image_resized * (1 - blurred_condition[..., np.newaxis]) + frame * blurred_condition[..., np.newaxis]).astype(np.uint8)
    ##########

    # Display the frame with the background changed
    cv2.imshow('Video with Background Changed', bg_image_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()