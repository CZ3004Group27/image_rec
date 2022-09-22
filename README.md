# How to call image rec fxns
1. unzip the file. your directory should look like this:
    ```
    - your folder
        - your file ...
        - your file ...
        - your file ...
        - final (the folder that you just extracted)
            - a lot of stuff but importantly
            - predict.py
            - detect.py
            - infer.py
            - ... and so on
    ```

2. navigate to `final` folder and run:
    ```python
    pip install -r requirements.txt
    ```
3. `predict.py` contains the fxn that can draw bounding boxes on images and save these outputs to `./final/run/detect/exp{i}` folder, where `i` represents the number of times the predict is called. This is suitable for week 8 task. To include the function that can draw bounding box, put the following in your code:
    ```python
    img_path = 'path_to_image'
    os.system(f'python ./final/predict.py {img_path}')
    ```
    If you want to predict > 1 photo, just put the `path_to_image` as the path to folder containing all the images that you want to predict.

4. `infer.py` contains the fxn that can infer what the object is without drawing bounding boxes. This is much faster and suitable for week 9 task. To include this function, put the following in your code:
    ```python
    img_path = 'path_to_image'
    res = os.popen(f'python ./final/infer.py {img_path}').read()
    print('result is: ', res)
    ```