import requests
import time


def generate_background(width, height):#max width or height is 1024
    url = "https://api.prodia.com/v1/sd/generate"

    payload = {
        "model": "v1-5-pruned-emaonly.safetensors [d7049739]",
        "prompt": "classroom, realistic",
        "negative_prompt": "badly drawn",
        "steps": 20,
        "cfg_scale": 7,
        "seed": -1,
        "upscale": False,
        "sampler": "DPM++ 2M Karras",
        "width": width,
        "height": height
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
        print(job_status)

    # Get the generated image URL
    image_url = response.json()['imageUrl']
    print(image_url)

    # Download the image
    image_response = requests.get(image_url)

    # Save the image as a .jpg file
    with open('zoom_background.jpg', 'wb') as f:
        f.write(image_response.content)

generate_background(1000,100)