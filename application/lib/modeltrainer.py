# import dependencies
from lib.image_preprocessor import ImagePreprocessor

import numpy as np
import os
from glob import glob

from keras.applications.inception_v3 import InceptionV3

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

from sklearn.metrics import f1_score
import re

# the ModelTrainer class
class ModelTrainer:
    # initialize default values
    def __init__(self):
        self.augmentations = 25
        self.epochs = 50
        self.batch_size = 50
        self.model_dir = 'saved_models'
        self.model_name = 'cnn_image_model.hdf5'
        self.root_dir = 'images'
        self.originals_dir = os.path.join(self.root_dir, 'original')
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.val_dir = os.path.join(self.root_dir, 'validation')
        self.test_dir = os.path.join(self.root_dir, 'test')
        self.random_seed = 7
        self.target_imagesize = (256, 256)
        self.categories = []
        self.training_count = 0
        self.validation_count = 0
        self.test_count = 0
        self.patience = 20
        self.clear = True
        self.test_size = 0.2
        self.val_size = 0.2
        
    # updates any default values and runs the functions to train the model
    def initialize(self):
        self.preprocessor = ImagePreprocessor()
        self.preprocessor.root_dir = self.root_dir
        self.preprocessor.originals_dir = self.originals_dir
        self.preprocessor.train_dir = self.train_dir
        self.preprocessor.val_dir = self.val_dir
        self.preprocessor.test_dir = self.test_dir
        self.preprocessor.test_size = self.test_size
        self.preprocessor.val_size = self.val_size
        self.preprocessor.clear = self.clear
        self.preprocessor.random_seed = self.random_seed
        
        self.preprocessor.initialize()
        self.categories = self.preprocessor.categories
        self.training_count = self.preprocessor.training_count
        self.validation_count = self.preprocessor.validation_count
        self.test_count = self.preprocessor.validation_count
        
        # Setting up the ImageDataGenerator for augmentation
        self.img_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rescale=1./255,
            fill_mode='reflect')
        
        self.train_generator = self.img_datagen.flow_from_directory(self.train_dir,
                                                           target_size=self.target_imagesize,
                                                           batch_size=self.augmentations,
                                                           shuffle=True,
                                                           seed=self.random_seed)

        self.validation_generator = self.img_datagen.flow_from_directory(self.val_dir,
                                                           target_size=self.target_imagesize,
                                                           batch_size=self.augmentations,
                                                           shuffle=True,
                                                           seed=self.random_seed)
        # If the validation accuracy doesn't improve 20 times in a row, training will stop
        self.earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=1, mode='auto')

        # If the validation accuracy improves, the model and its weights will be saved for later use
        self.checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_dir, self.model_name), 
                               verbose=1, save_best_only=True)
                               
        model = InceptionV3(include_top=False, weights = 'imagenet', input_shape = (self.target_imagesize[0], self.target_imagesize[1], 3))
        for layer in model.layers:
            layer.trainable = False

        # custom Layers 
        cus_layers = model.output
        cus_layers = Flatten()(cus_layers)

        cus_layers = Dense(256, activation="relu")(cus_layers)
        cus_layers = Dropout(0.4)(cus_layers)
        predictions = Dense(len(self.categories), activation="sigmoid")(cus_layers)

        model_final = Model(inputs = model.input, outputs = predictions)
        model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])
        # train the model
        model_final.fit_generator(self.train_generator,
                                 steps_per_epoch=self.training_count // 10, 
                                  epochs=self.epochs,
                                 validation_data = self.validation_generator,
                                 validation_steps=self.validation_count // 10,
                                 callbacks=[self.checkpointer, self.earlystopping],
                                 verbose=1)
        
        model_final.load_weights(filepath=os.path.join(self.model_dir, self.model_name))
        
        f1_score = self.__f1_score_cal(model_final)
        print("Model finished training")
        print("F1 Score: ", f1_score)
        
        # write the categories to file so they can be referrenced later by other scripts
        file = open(os.path.join(self.model_dir, self.model_name.replace(".hdf5", ".txt")),"w") 
 
        for category in self.categories:
            file.write(str(category) + "\n")
         
        file.close() 
        
        return
    
    # Takes an image location and returns a prediction on its category            
    def __predict_category(self, img_path, pred_model):
        img_tensor = self.preprocessor.file_to_tensor(img_path)
        h = pred_model.predict(img_tensor)
        return self.categories[np.argmax(h)]
        
    # Takes a model and calculates its F1 Score
    def __f1_score_cal(self, model):
        test_images = np.array(glob(os.path.join(self.test_dir, "*", "*")))
        y_true = []
        y_pred = []
        for img in test_images:
            y_true.append(self.categories.index(re.split(r'[\\/]',img)[-2]))
            pred = self.__predict_category(img, model)
            y_pred.append(self.categories.index(pred))
        
        return f1_score(y_true, y_pred, average='weighted')