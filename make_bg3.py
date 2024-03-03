import numpy as np
import base64
import requests
import time
import cv2
from PIL import Image
import io



def inpaint_image(img_extended, mask_data):
    # Convert the extended image to a NumPy array
    img_array = np.array(img_extended)
    # Encode the image data to base64
    imageData = base64.b64encode(img_array).decode()
    
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

def extend_image(image_path, additional_width):
    # Open the image file
    img = Image.open(image_path)
    width, height = img.size

    # Create an extension of the same height but with the specified additional width
    extension = Image.new('RGB', (additional_width, height), color='black')

    # Create a mask for the extension
    mask = Image.new('L', extension.size, color=255)

    # Concatenate the original image and the extension
    img_extended = Image.new('RGB', (width + additional_width, height))
    img_extended.paste(img, (0, 0))
    img_extended.paste(extension, (width, 0), mask)

    return img_extended

def create_mask(image, additional_width):
    # Get the dimensions of the image
    width, height = image.size

    # Create a mask for the original image (white)
    original_mask = Image.new('L', (width - additional_width, height), color=255)

    # Create a mask for the extension (black)
    extension_mask = Image.new('L', (additional_width, height), color=0)

    # Concatenate the original mask and the extension mask
    mask = Image.new('L', (width, height))
    mask.paste(original_mask, (0, 0))
    mask.paste(extension_mask, (width - additional_width, 0))

    # Convert the mask to a NumPy array
    mask_array = np.array(mask)

    # Save mask as image
    cv2.imwrite("mask.png", mask_array)

    # Encode the mask data to base64
    mask_base64 = base64.b64encode(mask_array).decode()

    return mask_base64

filename = 'avemichharden.jpg'
# Extend the image
extended_img = extend_image(filename, 300)

# Create a mask
mask = create_mask(extended_img, 300)

# Inpaint the image using the generated mask
inpaint_image(extended_img, mask)
