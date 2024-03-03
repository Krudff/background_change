import cv2
import mediapipe as mp
import numpy as np
import base64
import requests
import time

# Initialize the selfie segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 1)

def generate_mask(image_path):
    # Load an image
    image = cv2.imread(image_path)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the face detections
    results = mp_selfie_segmentation.process(image_rgb)

    # Create an empty mask with the same shape as the frame
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Get the condition where segmentation mask is 1 (i.e., person is present)
    condition = results.segmentation_mask > 0.05

    # Apply the condition to the frame (make it white)
    mask[~condition] = 255

    # Save mask as image
    cv2.imwrite("mask.png", mask)

    # Convert the mask data to base64
    mask_data = base64.b64encode(cv2.imencode('.png', mask)[1]).decode()

    return mask_data

def inpaint_image(image_path, mask_data):
    # Read the image file and convert it to base64
    with open(image_path, 'rb') as f:
        imageData = base64.b64encode(f.read()).decode('utf-8')

    url = "https://api.prodia.com/v1/sd/inpainting"

    payload = {
        "imageData": imageData,
        "maskData": mask_data,
        "model": "v1-5-pruned-emaonly.safetensors [d7049739]",
        "prompt": "classroom",
        "negative_prompt": "badly drawn",
        "steps": 20,
        "cfg_scale": 7,
        "seed": -1,
        "upscale": False,
        "mask_blur": 4,
        "inpainting_fill": 1,
        "inpainting_mask_invert": 1,
        "inpainting_full_res": False,
        "sampler": "DPM++ 2M Karras",
        "width": 1024,
        "height": 576
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-Prodia-Key": "7127d01a-d0c9-435a-ad98-31179a49071f"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)
    print(response.json()['job'])

    job_id = response.json()['job']

    # Poll the API until the job is done
    while True:
        response = requests.get(f'https://api.prodia.com/v1/job/{job_id}', headers=headers)
        job_status = response.json()['status']
        if job_status == 'succeeded':
            break
        time.sleep(1)  # wait for a second before polling again
        print("Debugger: ",job_status)

    # Get the generated image URL
    image_url = response.json()['imageUrl']
    print(image_url)

    # Download the image
    image_response = requests.get(image_url)

    # Save the image as a .jpg file
    with open('inpainting_result.jpg', 'wb') as f:
        f.write(image_response.content)


filename = 'avemichharden.jpg'
# Generate the mask
mask_data = generate_mask(filename)

# Inpaint the image using the generated mask
inpaint_image(filename, mask_data)
