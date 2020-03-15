### Complete credit goes to David Sandberg, original code found: https://github.com/davidsandberg/facenet

Additional Ref (how to use the repo):
-  https://medium.com/@nhbond/how-to-copying-what-david-sandberg-did-4bf6e0ad3cd5 

### My use case:
-  Augment training data, found in `./src/new_contrib.py`
-  Train face identifier model

### Quicktart:
-  Install requirements `python3 -m pip install -r requirements.txt`
-  To train your own data, follow the folder structure below and place images
accordingly
-  To use a pretrained model, make a models directory and install model
    -  `mkdir ./models`
    -  

### Folder Structure:
```
-  data
    -  images
        -  train_raw
            -  person1_name
                -  example_image.jpg
                -  img2.jpg
            -  person2_name
                -  another_ex_image.jpg
        -  test_raw
            -  person1_name
                -  tets_im.jpg
            -  person2_name
                -  ex.jpg
```