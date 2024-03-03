import cv2
import mediapipe as mp
import numpy as np

# Initialize the selfie segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation()
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


# Open two video files
cap1 = cv2.VideoCapture('vid1.mp4')
cap2 = cv2.VideoCapture('vid2.mp4')

# Open background image
image = cv2.imread('long.png')
bg_image = cv2.resize(image, (1280, 480))
#####

while True:
    #ret, frame = cap.read()
    #if not ret:
    #    break

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break


    # Resize the frames
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    
    # Convert the frame to RGB
    frame_rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Process the frame
    results1 = mp_selfie_segmentation.process(frame_rgb1)
    results2 = mp_selfie_segmentation.process(frame_rgb2)
    # Process the frame for hand tracking
    hand_results1 = mp_hands.process(frame_rgb1)
    hand_results2 = mp_hands.process(frame_rgb2)


    # Create an empty mask with the same shape as the frame
    mask1 = np.zeros(frame1.shape, dtype=np.uint8)
    mask2 = np.zeros(frame2.shape, dtype=np.uint8)

    # Get the condition where segmentation mask is 1 (i.e., person is present)
    condition1 = results1.segmentation_mask > 0.1
    condition2 = results2.segmentation_mask > 0.1

    # Get the condition where hands are present
    if hand_results1.multi_hand_landmarks:
        for hand_landmarks in hand_results1.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame1.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                condition1[cy][cx] = True
    if hand_results2.multi_hand_landmarks:
        for hand_landmarks in hand_results2.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame2.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                condition2[cy][cx] = True

    # Apply the condition to the frame
    mask1[condition1] = frame1[condition1]
    mask2[condition2] = frame2[condition2]

    # Resize the background image to match the frame size
    bg_image1 = bg_image[:, :640]
    bg_image2 = bg_image[:, 640:]
    bg_image_resized1 = cv2.resize(bg_image1, (frame1.shape[1], frame1.shape[0]))
    bg_image_resized2 = cv2.resize(bg_image2, (frame2.shape[1], frame2.shape[0]))

    ##########
    # Apply the condition to the background image
    #bg_image_resized[condition] = frame[condition]

    # Create a blurred version of the condition #(25,25) is the kernel size (less will make it sharper)
    blurred_condition1 = cv2.GaussianBlur(condition1.astype(np.float32), (25, 25), 0)
    blurred_condition2 = cv2.GaussianBlur(condition2.astype(np.float32), (25, 25), 0)

    # Apply the blurred condition to the background image
    bg_image_resized1 = (bg_image_resized1 * (1 - blurred_condition1[..., np.newaxis]) + frame1 * blurred_condition1[..., np.newaxis]).astype(np.uint8)
    bg_image_resized2 = (bg_image_resized2 * (1 - blurred_condition2[..., np.newaxis]) + frame2 * blurred_condition2[..., np.newaxis]).astype(np.uint8)
    ##########

    # Concat the two frames
    #frame = np.concatenate((frame1, frame2), axis=1)
    concatenated_frame = np.concatenate((bg_image_resized1, bg_image_resized2), axis=1)
    # Display the frame with the background changed
    cv2.imshow('Video with Background Changed', concatenated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
mp_hands.close()
cap1.release()
cap2.release()
cv2.destroyAllWindows()