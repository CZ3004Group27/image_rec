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
    category = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'S', 'T', 'U', 'V', 'W',
                'X', 'Y', 'Z', 'Bullseye', 'Down', 'Eight', 'Five', 'Four', 'Left', 'Nine', 'One', 'Right', 'Seven', 'Six', 'Stop', 'Three', 'Two', 'Up']

    output = model(img).pred
    res = []
    for i in range(len(output[0])):
        res.append(category[int(output[0][i][-1])])
    # print(float(output[0][0][4])) # conf score
    if len(output[0]) == 0:
        return 'Nothing detected'
    return res

# unittest, or when running this file using `os.popen(python...)`
if __name__ == "__main__":
    image_path = sys.argv[1]
    img = get_image_from(image_path)
    res = infer(img)
    print(res, end="") # do not add \n to result
