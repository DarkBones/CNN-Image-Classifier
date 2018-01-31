import os
import numpy as np
from glob import glob
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import re
import random
import shutil

class ImagePreprocessor:
    def __init__(self):
        self.root_dir = os.path.join('..', 'application', 'images')
        self.originals_dir = os.path.join(self.root_dir, 'original')
        self.training_dir = os.path.join(self.root_dir, 'train')
        self.val_dir = os.path.join(self.root_dir, 'validation')
        self.test_dir = os.path.join(self.root_dir, 'test')
        
        self.target_imagesize = (256, 256)
        
        # the sizes of the test and validation sets as compared to the total amount of images
        self.test_size = 0.2
        self.validation_size = 0.2
        
        self.clear_existing_data = False # if true, data in training, test and validation directories will be deleted before splitting the data in the originals directory
        
        self.random_seed = 7
        
        # the amount of images in the training, validation and test sets
        self.training_count = 0
        self.validation_count = 0
        self.test_count = 0
        
        # list of categories
        self.categories = []
        
    def initialize(self):
        self.__split_dataset()
        
        self.training_count, self.validation_count, self.test_count = self.__get_dataset_sizes()
        self.categories = self.__get_categories()
        
        # print statistics about the dataset
        print('%d image categories' % len(self.categories))
        print('%d total images' % sum([self.training_count, self.validation_count, self.test_count]))
        print("")
        print('%d training images' % self.training_count)
        print('%d validation images' % self.validation_count)
        print('%d test images'% self.test_count)

        print("")
        print("Categories:")
        for c in self.categories:
            print("  - %s" % c)
        
        return
    
    # function to remove all files in given directory
    def __empty_directory(self, path):
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        return
    
    # splits images in original directory into training, test and validation directories
    def __split_dataset(self):
        random.seed(self.random_seed)
        
        size_count = 0
        for category in os.listdir(self.originals_dir):
            # make a new directory where they don't exist and empty existing directories
            for p in [re.split(r'[\\/]', self.training_dir)[-1], re.split(r'[\\/]', self.val_dir)[-1], re.split(r'[\\/]', self.test_dir)[-1]]:
                if not os.path.exists(os.path.join(self.root_dir, p, category)):
                    os.makedirs(os.path.join(self.root_dir, p, category))
                if self.clear_existing_data == True:
                    self.__empty_directory(os.path.join(self.root_dir, p, category))
                
            # collect all the files in the originals directory
            files = []
            for file in os.listdir(os.path.join(self.originals_dir, category)):
                files.append(file)
            
            # calculate the training, validation and test set sizes
            test_count = round(len(files) * self.test_size)
            validation_count = round(len(files) * self.validation_size)
            train_count = len(files) - test_count - validation_count
            
            # randomly shuffle the array of files
            random.shuffle(files)
            
            for i, file in enumerate(files):
                location = None
                if i < test_count:
                    location = self.test_dir
                elif i < test_count + validation_count:
                    location = self.val_dir
                else:
                    location = self.training_dir
                    
                shutil.copyfile(os.path.join(self.originals_dir, category, file), os.path.join(location, category, file))
        return
    
    # load file locations and labels
    def load_dataset(self, path):
        data = load_files(path)
        files = np.array(data['filenames'])
        targets = np_utils.to_categorical(np.array(data['target']), max(data['target'])+1)
        return files, targets
    
    # takes an image filepath and returns its tensor representation
    def file_to_tensor(self, file):
        img = image.load_img(file, target_size=self.target_imagesize)
        x = image.img_to_array(img)
        x *= (1.0/x.max()) # set the range of the tensor values between 0 and 1
        return np.expand_dims(x, axis=0)
    
    # takes a list of image filepaths and returns a list of 4D tensors
    def files_to_tensors(self, files):
        list_of_tensors = [self.file_to_tensor(file) for file in files]
        return np.vstack(list_of_tensors)
    
    # returns an array with the category names
    def __get_categories(self):        
        return [item[len(self.originals_dir)+1:] for item in sorted(glob(os.path.join(self.originals_dir, "*")))]
    
    # returns the sizes of the training, validation and test sets
    def __get_dataset_sizes(self):
        train_size = sum([len(files) for r, d, files in os.walk(self.training_dir)])
        val_size = sum([len(files) for r, d, files in os.walk(self.val_dir)])
        test_size = sum([len(files) for r, d, files in os.walk(self.test_dir)])
        
        return train_size, val_size, test_size