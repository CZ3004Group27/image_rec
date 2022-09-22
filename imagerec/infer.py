# #You may need to install these packages first
import torch
from imagerec.models.experimental import attempt_load
import numpy as np
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import time
import sys

best_pt = './final/weights/best.pt'
model = torch.hub.load("./final", 'custom', path=best_pt, source='local') # local model dont need internet

#Function for single prediction
def infer():
    # Do not touch the categories
    category = ['Alphabet A', 'Alphabet B', 'Alphabet C', 'Alphabet D', 'Alphabet E', 'Alphabet F', 'Alphabet G', 'Alphabet H', 'Alphabet S', 'Alphabet T', 'Alphabet U', 'Alphabet V', 'Alphabet W',
                'Alphabet X', 'Alphabet Y', 'Alphabet Z', 'Bulls Eye', 'Down Arrow', 'Eight', 'Five', 'Four', 'Left Arrow', 'Nine', 'One', 'Right Arrow', 'Seven', 'Six', 'Stop', 'Three', 'Two', 'Up Arrow']

    image = sys.argv[1]
    img = Image.open(image)
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
        print(category[int(output[index][0][-1])]) 
        return category[int(output[index][0][-1])]
    # print(float(output[0][0][4])) # conf score
    print(category[int(output[0][0][-1])]) 
    return category[int(output[0][0][-1])]

res = infer()


    
