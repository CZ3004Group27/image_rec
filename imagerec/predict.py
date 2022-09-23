import os
from PIL import Image
import sys

def merge_image():
    predicted_folder = './final/runs/detect/'
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

def predict():
    weight_path = './final/weights/best.pt'
    image_path = sys.argsv[1]
    os.system(f'python ./final/detect.py --weights {weight_path} --img 640 --source {image_path}')
    merge_image()

predict()