import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()

# Load the video
cap = cv2.VideoCapture('vid2.mp4')

# Load the background image
background = Image.open('sajang.png')

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the background image to match the frame size
    background_resized = cv2.resize(np.array(background), (frame.shape[1], frame.shape[0]))

    # Convert the background image to BGR format
    background_resized = cv2.cvtColor(background_resized, cv2.COLOR_RGB2BGR)

    # Convert the frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply the transformations needed
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    # Make sure the model is in evaluation mode
    model.eval()

    # Predict the segmentation mask
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create a color palette for visualization
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Plot the semantic segmentation predictions
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(pil_image.size)
    r.putpalette(colors)

    # Convert the mask to numpy array
    mask = np.array(r)

    # Create a 3D mask for the color channels
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Replace the background of the frame
    frame[mask_3d == 0] = background_resized[mask_3d == 0]

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
