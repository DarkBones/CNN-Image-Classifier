# CNN-Image-Classifier

The purpose of this application is to train a classifier to categorize images and sort these images into different subdirectories accordingly. Depending on the way you set up your dataset, pictures of your face go in the 'myface' directory, pictures of your dog go in the 'mydog' directory, and pictures of landscapes go in the 'landscapes' directory.

This makes managing the hundreds of pictures that live in the DCIM directory on your smartphone a lot easier.

# How To Use
There are two main components to this application:
- train.py
- process.py

## Train
The code in *train.py* trains a *Convolutional Neural Network* (CNN) to recognize images in your dataset. Your dataset must be structured as follows:
- images
  - A directory named 'original'
    - category1
      - image1_category_1.jpg
      - image2_category_1.jpg
    - category2
      - image1_category2.jpg
      - image2_category2.jpg
      
You can have as many of these categories as you choose, although it's recommended that you have at least **100 images per category**. The sample dataset provided is structured as follows:
- images
  - original
    - animal
    - city_scape
    - food
    - group
    - landscape
    - me

### Optional Parameters
Parameter|Description|Default Value|
-|-|-
-model_dir|Directory where the model will be saved to|saved_models
-model_name|The name of the finished model (will automatically append '.hdf5' is missing)|cnn_image_model.hdf5
-data_dir|The root directory of the image dataset|images
-clear|If set to 'True', all instances in the training, validation, and test data will be cleared before training the model|True
-augmentations|How many times each image should be augmented|25
-patience|How many epochs without improvement on validation_loss should occur before the model stops training|20
-epochs|The maximum amount of epochs to train the model on|500

### Example
#### Example:
>train.py -clear False -augmentations 5 model_name new_model

The example above will not delete any existing data in the training, validation and test sets, it will train the model using 5 augmentations for each image and it will save the model as *new_model.hdf5*.

**NOTE:** After the model has been trained, the script will also save a text file with the same name and in the same location as the model containing all the categories. This file can later be read by *process.py* so it knows what categories to use.

## Process
When the model has finished training, you can run *process.py* to start organizing your photos.
### Optional parameters:

Parameter|Description|Default Value
-|-|-
-model_dir|Location of the model we trained using *train.py* earlier|saved_models
-model_name|The name of the trained model (will automatically append '.hdf5' is missing)|cnn_image_model.hdf5
-image_dir|The location of the images the model needs to classify|test_images
-threshold|How confident the model needs to be in order to classify an image as a category|0.1
-copy|If set to 'True', it will keep the original images after they have been classified|True

When *process.py* is run, it will create subdirectories for each of the categories and copy the original images to them based on the model's predictions and the processor's threshold.

## Dependencies
The following modules must be installed in order for this application to work:
- Python 3.x
- keras
- tensorflow
- numpy
