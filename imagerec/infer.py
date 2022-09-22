# #You may need to install these packages first
import torch
from imagerec.models.experimental import attempt_load
from imagerec import weights
import imagerec
import numpy as np
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import time
import sys
from imagerec.helpers import get_path_to

# Set all the parameters here
best_pt_filename = "best.pt"

model_weights_folder = get_path_to(weights)
imagerec_path = get_path_to(imagerec)

best_pt_path = model_weights_folder.joinpath(best_pt_filename)
model = torch.hub.load(str(imagerec_path), 'custom', path=str(best_pt_path), source='local') # local model dont need internet

def get_image_from(image_path: str):
    return Image.open(image_path)

#Function for single prediction
def infer(img) -> str:
    # Do not touch the categories
    category = ['Alphabet A', 'Alphabet B', 'Alphabet C', 'Alphabet D', 'Alphabet E', 'Alphabet F', 'Alphabet G', 'Alphabet H', 'Alphabet S', 'Alphabet T', 'Alphabet U', 'Alphabet V', 'Alphabet W',
                'Alphabet X', 'Alphabet Y', 'Alphabet Z', 'Bulls Eye', 'Down Arrow', 'Eight', 'Five', 'Four', 'Left Arrow', 'Nine', 'One', 'Right Arrow', 'Seven', 'Six', 'Stop', 'Three', 'Two', 'Up Arrow']

    output = model(img).pred
    if len(output) > 1:
        max_area, index = 0, 0
        for i, op in enumerate(output):
            length = op[0][2] - op[0][0]
            width = op[0][3] - op[0][1]
            area = length*width
            if area > max_area:
                max_area = area
                index = i
        return category[int(output[index][0][-1])]
    # print(float(output[0][0][4])) # conf score
    return category[int(output[0][0][-1])]

# unittest, or when running this file using `os.popen(python...)`
if __name__ == "__main__":
    image_path = sys.argv[1]
    img = get_image_from(image_path)
    res = infer(img)
    print(res, end="") # do not add \n to result
