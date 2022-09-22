# How to call image rec package

## Step 1: Install the package in development mode

Before you start, it is better to create a virtual environment
for installation, just in case there is any dependency conflict.
In the root folder containing `setup.py` file, run

```sh
pip install -e .
```

This will make the package `imagerec` available for imports.

## Step 2: Note the important scripts

1. The main package should look like this:

    ```markdown
        - imagerec (the folder that you just extracted)
            - a lot of stuff but importantly
            - predict.py
            - detect.py
            - infer.py
            - ... and so on
    ```

2. `predict.py` contains the fxn that can draw bounding boxes on images and
   save these outputs to `./final/run/detect/exp{i}` folder, where `i`
   represents the number of times the predict is called. This is suitable for
   week 8 task. To include the function that can draw bounding box, put the
   following in your code:

    ```python
    img_path = 'path_to_image'
    os.system(f'python ./final/predict.py {img_path}')
    ```

    If you want to predict > 1 photo, just put the `path_to_image` as the path to folder containing all the images that you want to predict.

3. `infer.py` contains the fxn that can infer what the object is without
   drawing bounding boxes. This is much faster and suitable for week 9 task.

    To run this function in CLI within python, put the following in your code:

    ```python
    import os
    img_path = 'path_to_image'
    res = os.popen(f'python -m imagerec.infer {img_path}').read()
    print('result is: ', res)
    ```

    Alternatively, to run the code in the usual way:

    ```python
    from imagerec.infer import infer, get_image_from
    image_path = 'path_to_image'
    image = get_image_from(image_path)
    res = infer(image)
    print('result is: ', res)
    ```
