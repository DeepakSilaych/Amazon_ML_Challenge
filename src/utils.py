import os
import requests
from PIL import Image
from io import BytesIO

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except IOError as e:
        print(f"Error saving image to {save_path}: {e}")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
