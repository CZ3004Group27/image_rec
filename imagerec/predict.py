import os
from PIL import Image
import imagerec.runs.detect
from imagerec import weights
import time
import sys

from imagerec.helpers import get_path_to

# Set up all the parameters here
best_pt_filename = "best.pt"

detect_folder = get_path_to(imagerec.runs.detect)
model_weights_folder = get_path_to(weights)
best_pt_path = model_weights_folder.joinpath(best_pt_filename)

def merge_image():
    predicted_folder = str(detect_folder)
    predicted_folder = predicted_folder+os.listdir(predicted_folder)[-1] + '/'
    images_names = os.listdir(predicted_folder)

    images = []
    for imageName in images_names:
        imageLocation = predicted_folder + imageName
        img = Image.open(imageLocation)
        images.append(img)
    
    size = len(images)
    width = images[0].width
    height = images[0].height
    merged_image = Image.new('RGB', (size*width, height))
    for i in range(size):
        merged_image.paste(images[i], (i*width,0))

    save_path = predicted_folder + 'merged_image.jpg'
    merged_image.save(save_path)

def predict(image_path):
    weight_path = str(best_pt_path)
    os.system(f'python -m imagerec.detect --weights {weight_path} --img 640 --source {image_path}')
    merge_image()

if __name__ == "__main__":
    image_path = sys.argv[1]
    res = predict(image_path)
    print(res, end="")
