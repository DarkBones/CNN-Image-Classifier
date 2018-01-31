"""
This script takes a trained model in hdf5 format, along with its categories in a separate .txt file, and categorizes images found in a specified directory.
It will use the model to predict what category an image should be and it will copy the image file to the appropriate subdirectorie. If the image contains multiple categories, it will copy the image to multiple subdirectories.
"""

# import dependencies
from lib.image_preprocessor import ImagePreprocessor
import numpy as np
import os
from keras.models import Sequential, Model
from keras.models import load_model
import shutil

# the ImageProcessor class
class ImageProcessor:
    # instantiate default values
    def __init__(self):
        self.model_dir = 'saved_models'
        self.model_name = 'cnn_image_model.hfd5'
        self.image_dir = 'images_test'
        self.categories = []
        self.threshold = 0.1
        self.copy = True
        self.model = None
        
    def initialize(self):
        # read the .txt file containing the categories and save each line to the categories variable
        with open(os.path.join(self.model_dir, self.model_name.replace(".hdf5", ".txt"))) as fp:  
            line = fp.readline()
            while line:
                self.categories.append(line.strip())
                line = fp.readline()
        
        # instantiate the ImagePreprocessor
        self.preprocessor = ImagePreprocessor()
        
        # create subdirectories for each image category
        self.__create_subdirs(self.image_dir, self.categories)
        
        # load the classification model
        self.model = load_model(os.path.join(self.model_dir, self.model_name))
        
        # categorize the images
        self.__categorize_images(self.image_dir, self.copy, self.model, self.preprocessor, self.threshold, self.categories)
        
        return
        
    # creates subdirectories for each entry in list 'subdirs'
    def __create_subdirs(self, root, subdirs):
        for subdir in subdirs:
            if os.path.isdir(os.path.join(root, subdir)) == False:
                os.makedirs(os.path.join(root, subdir))
        return
    
    # iterates through all images in 'root' and classifies them
    def __categorize_images(self, root, copy, model, preprocessor, threshold, categories):
        for file in os.listdir(root):
            categorized = False
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png'):
                preds = self.__predict_category(os.path.join(root, file), model, preprocessor)
                for i, pred in enumerate(preds):
                    # if the prediction is higher than the threshold, copy the image to that subdirectory
                    if pred > threshold:
                        categorized = True
                        shutil.copyfile(os.path.join(root, file), os.path.join(root, categories[i], file))
            
            # if copy is false and a category has been found, remove the original image
            if copy == False and categorized == True:
                os.remove(os.path.join(root, file))
    
    # takes an image location and a model and returns an array of predictions
    def __predict_category(self, img_path, pred_model, preprocessor):
        img_tensor = preprocessor.file_to_tensor(img_path)
        h = pred_model.predict(img_tensor)
        return h[0]
        