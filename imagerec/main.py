import os

img_path = '../test_labels'
os.system(f'python predict.py {img_path}')
for img in os.listdir(img_path)[1:]:
    img = os.path.join(img_path, img)
    os.system(f'python infer.py {img}')
