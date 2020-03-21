### Complete credit goes to David Sandberg, original code found: https://github.com/davidsandberg/facenet

Additional Ref (how to use the repo):
-  https://medium.com/@nhbond/how-to-copying-what-david-sandberg-did-4bf6e0ad3cd5 

### My use case:
-  Augment training data, found in `./src/new_contrib/augment_data.py`
-  Train face identifier model
-  Problem: Require GPU? So try on Google Colab instead; document this code
and complete the augment but try POC on Colab

### Quicktart:
-  *Addition*: Augment images if so desired; adjust pathing in `./src/new_contrib/augment_data.py` and run;
can play with augmentations found in the doc
-  Install requirements `python3 -m pip install -r requirements.txt`
-  To train your own data, follow the folder structure below and place images
accordingly
-  To use a pretrained model, make a models directory and install model
    -  `mkdir ../../datas/models in desired location (have to modify the commands below accordingly)`
    -  Can install inception modle from the git ref link above; using the more powerful one into the models dir
    -  `export PYTHONPATH=$(pwd)/src`
    -  Align training and test images (requires GPU to run):
    ```
    python ./src/align/align_dataset_mtcnn.py \
    ../../data/facenet/images/train_augmented_isr \
    ../../data/facenet/images/train_aligned_isr \
    --image_size 160
    ```
    ```
    python ./src/align/align_dataset_mtcnn.py \
    ../../data/facenet/images/test_raw \
    ../../data/facenet/images/test_aligned \
    --image_size 160
    ```
    -  Train classifier on imagess
    ```
    python ./src/classifier.py TRAIN \
    ../../data/facenet/images/train_aligned_isr/ \
    ../../data/facenet/models/20180402-114759/20180402-114759.pb \
    ../../data/facenet/models/20180402-114759/my_classifier_isr.pkl
    ```
    -  Match images from test_aligned to people trained in classifier
    ```
    python ./src/classifier.py CLASSIFY \
    ../../data/facenet/images/test_aligned_isr/ \
    ../../data/facenet/models/20180402-114759/20180402-114759.pb \
    ../../data/facenet/models/20180402-114759/my_classifier_isr.pkl
    ```


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
-  main
    -  facenet
        - *The README.md you are reading right now*
```