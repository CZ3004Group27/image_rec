
# #You may need to install these packages first
import torch
from models.experimental import attempt_load
import numpy as np
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import time


# def preprocess(path):
#     img = cv2.imread(path,1)
    
#     img = cv2.resize(img, (416,416))
#     #brighten image
#     alpha = 1.2
#     beta = 80
#     brightened_img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype),0,beta)

#     nmax = 255
#     nmin = 0

#     brightened_img = cv2.normalize(brightened_img, None, alpha = nmin, beta = nmax, norm_type = cv2.NORM_MINMAX)
    
#     lab_img = cv2.cvtColor(brightened_img, cv2.COLOR_BGR2LAB)

#     l, a, b = cv2.split(lab_img)    

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     clahe_img = clahe.apply(l)

#     updated_image = cv2.merge((clahe_img,a,b))

#     CLAHE_img = cv2.cvtColor(updated_image, cv2.COLOR_LAB2BGR)

#     return CLAHE_img

#Function for single prediction
def infer():
    start = time.time()
    #To change path and image path
    
    # filename = img.split('/')[-1].split('.')[0]
    best_pt = './runs/train/exp/weights/best.pt'
    img = './test/image1.jpg'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=best_pt)

    # Do not touch the categories
    category = ['Alphabet A', 'Alphabet B', 'Alphabet C', 'Alphabet D', 'Alphabet E', 'Alphabet F', 'Alphabet G', 'Alphabet H', 'Alphabet S', 'Alphabet T', 'Alphabet U', 'Alphabet V', 'Alphabet W',
                'Alphabet X', 'Alphabet Y', 'Alphabet Z', 'Bulls Eye', 'Down Arrow', 'Eight', 'Five', 'Four', 'Left Arrow', 'Nine', 'One', 'Right Arrow', 'Seven', 'Six', 'Stop', 'Three', 'Two', 'Up Arrow']

    #Load the saved weights
    #To save time in the actual run, we can load a model outside the function so that weights do not have to be reinstantiated.
    img = Image.open(img)
    output = model(img).pred
    # output = int(output[0][0][-1])
    
    # print(output[0][0][-1])
    # annotate_img = Image.open(img)
    # output = category_map[int(category[output])]
    print(output[0][0][-1])
    print(category[int(output[0][0][-1])])

    print(time.time()-start)

    # return output


infer()


    
