import cv2
import mediapipe as mp
import numpy as np

def process_videos(video_files, bg_image_file):
    # Initialize the selfie segmentation model
    mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Open the video files
    caps = [cv2.VideoCapture(video_file) for video_file in video_files]

    # Open background image
    image = cv2.imread(bg_image_file)
    height, width = image.shape[:2]

    # Calculate the width for each video frame in the concatenated image
    # Here you can set a minimum width for each frame to ensure they are not too small
    min_frame_width = 160
    frame_width = max(width // len(video_files), min_frame_width)

    while True:
        frames = []
        masks = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to the calculated width while maintaining aspect ratio
            frame_height = int(frame.shape[0] * (frame_width / frame.shape[1]))
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = mp_selfie_segmentation.process(frame_rgb)

            # Create an empty mask with the same shape as the frame
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # Get the condition where segmentation mask is 1 (i.e., person is present)
            condition = results.segmentation_mask > 0.2

            # Apply the condition to the frame
            mask[condition] = 255

            frames.append(frame)
            masks.append(mask)

        # Ensure the background image is large enough to accommodate the frames
        bg_image = cv2.resize(image, (frame_width * len(video_files), frame_height))

        concatenated_frame = None
        for i in range(len(frames)):
            # Apply the condition to the background image
            bg_image_resized = bg_image[:, frame_width*i:frame_width*(i+1)]
            bg_image_resized[masks[i] == 255] = frames[i][masks[i] == 255]

            # Concat the frames
            if concatenated_frame is None:
                concatenated_frame = bg_image_resized
            else:
                concatenated_frame = np.concatenate((concatenated_frame, bg_image_resized), axis=1)

        # Display the frame with the background changed
        cv2.imshow('Video with Background Changed', concatenated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()

# Call the function with your video files and background image
process_videos(['vid2.mp4', 'vid2.mp4', 'vid2.mp4', 'vid2.mp4','vid2.mp4', 'vid2.mp4', 'vid2.mp4', 'vid2.mp4',], 'long.png')
