# import dependencies
from image_preprocessor import ImagePreprocessor

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

# setting the parameters
augmentations = 20 # how many augmentations to make for each original image

epochs = 500
batch_size = 50
saved_models_dir = "saved_models_test"

modelname = 'cnn_image_model'

preprocessor = ImagePreprocessor()
preprocessor.root_dir = 'images'
preprocessor.originals_dir = os.path.join(preprocessor.root_dir, 'original')
preprocessor.training_dir = os.path.join(preprocessor.root_dir, 'train')
preprocessor.test_dir = os.path.join(preprocessor.root_dir, 'test')
preprocessor.val_dir = os.path.join(preprocessor.root_dir, 'validation')
preprocessor.random_seed = 7
preprocessor.target_imagesize = (256, 256)
preprocessor.clear_existing_data = False # if true, data in training, test and validation directories will be deleted before splitting the data in the originals directory

preprocessor.initialize()
categories = preprocessor.categories
training_count = preprocessor.training_count
validation_count = preprocessor.validation_count
test_count = preprocessor.test_count

# Setting up the ImageDataGenerator for augmentation
img_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255,
    fill_mode='reflect')

train_generator = img_datagen.flow_from_directory(preprocessor.training_dir,
                                                   target_size=preprocessor.target_imagesize,
                                                   batch_size=augmentations,
                                                   shuffle=True,
                                                   seed=preprocessor.random_seed)

validation_generator = img_datagen.flow_from_directory(preprocessor.val_dir,
                                                   target_size=preprocessor.target_imagesize,
                                                   batch_size=augmentations,
                                                   shuffle=True,
                                                   seed=preprocessor.random_seed)
                                                   
# If the validation accuracy doesn't improve 20 times in a row, training will stop
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

# If the validation accuracy improves, the model and its weights will be saved for later use
checkpointer = ModelCheckpoint(filepath=os.path.join(saved_models_dir, modelname + '.hdf5'), 
                       verbose=1, save_best_only=True)
            
# Takes an image location and returns a prediction on its category            
def predict_category(img_path, pred_model):
    img_tensor = preprocessor.file_to_tensor(img_path)
    h = pred_model.predict(img_tensor)
    return categories[np.argmax(h)]
    
# Takes a model and calculates its F1 Score
def f1_score_cal(model):
    test_images = np.array(glob(os.path.join(preprocessor.test_dir, "*", "*")))
    y_true = []
    y_pred = []
    for img in test_images:
        y_true.append(categories.index(re.split(r'[\\/]',img)[-2]))
        pred = predict_category(img, model)
        y_pred.append(categories.index(pred))
    
    return f1_score(y_true, y_pred, average='weighted')
    
print("Training ", modelname)

model = InceptionV3(include_top=False, weights = 'imagenet', input_shape = (preprocessor.target_imagesize[0], preprocessor.target_imagesize[1], 3))
for layer in model.layers:
    layer.trainable = False

# custom Layers 
cus_layers = model.output
cus_layers = Flatten()(cus_layers)

cus_layers = Dense(256, activation="relu")(cus_layers)
cus_layers = Dropout(0.4)(cus_layers)
predictions = Dense(len(categories), activation="softmax")(cus_layers)

model_final = Model(inputs = model.input, outputs = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])
# train the model
model_final.fit_generator(train_generator,
                         steps_per_epoch=training_count // 10, 
                          epochs=epochs,
                         validation_data = validation_generator,
                         validation_steps=validation_count // 10,
                         callbacks=[checkpointer, earlystopping],
                         verbose=1)

f1_score = f1_score_cal(model_final)
print("Model finished training")
print("F1 Score: ", f1_score)