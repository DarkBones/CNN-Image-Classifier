# import dependencies
import sys
from sys import argv
import os
import re

# instantiate default values
model_dir = 'saved_models'
model_name = 'cnn_image_model.hdf5'
image_dir = 'test_images'
threshold = 0.1
copy = True

# prints instructions to the console
def print_help():
    print("------------- HOW TO USE -------------")
    print("")
    print("This script takes a trained Convolutional Neural Network and runs through a set of images to classify them.")
    print("")
    print("ARGUMENT\tDESCRIPTION\t\t\t\tDEFAULT VALUE")
    print('-model_dir\tLocation of the trained model\t\tsaved_models')
    print('-model_name\tThe name of the trained model\t\tcnn_image_model.hdf5 (will append .hdf5 automatically)')
    print('-image_dir\tLocation of unclassified images\t\ttest_images')
    print('-threshold\tClassification threshold of the model\t0.1')
    print('-copy\t\tIf true, copy images instead of moving\tTrue')
    
# returns given command line arguments and returns them as a dictionary
def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            value = ''
            if len(argv) > 1:
                value = argv[1]
            opts[argv[0]] = value # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts
    
# takes a string and tries to parse it into a float
def parse_float(str):
    float_format = re.compile(r'\d*\.\d+')
    if len(re.match(float_format, str)[0]) == len(str):
        str_spl = re.split(r'\.', str)
        if len(str_spl[0]) == 0:
            str_spl[0] = '0'
        if len(str_spl[1]) == 0:
            str_spl[1] = '0'
        
        bef = int(str_spl[0])
        aft = int(str_spl[1])
        aft /= 10 ** len(str_spl[1])
        return bef + aft
    return
    
# aborts the script and prints an error message if one is given
def abort(error=None):
    print('APPLICATION ABORTED')
    if error != None:
        print('ERROR:')
        print(error)
    exit()
    
# parses the arguments given in the command and changes the default parameters
def parse_args(args):
    global model_dir, model_name, image_dir, threshold, copy
    
    if '-model_dir' in args:
        model_dir = args['-model_dir']
    if '-model_name' in args:
        model_name = args['-model_name']
        if model_name.endswith('.hdf5') == False:
            model_name = str(model_name) + '.hdf5'
    if '-image_dir' in args:
        image_dir = args['-image_dir']
    if '-threshold' in args:
        threshold = parse_float(args['-threshold'])
        if threshold == None:
            abort('Incorrect value for \'-threshold\'. Expected float. Received: \'' + args['-threshold'] + '\'')
        elif threshold >= 1:
            abort('Incorrect value for \'-threshold\'. Expected float between 0 and 1. Received: \'' + str(threshold) + '\'')
    if '-copy' in args:
        if args['-copy'].lower() in ['true', 'yes', '1', 'y', 't']:
            copy = True
        elif args['-copy'].lower() in ['false', 'no', '0', 'n', 'f']:
            copy = False
        else:
            abort('Incorrect value for \'-copy\'. Expected \'True\' or \'False\'. Received: \'' + args['-copy'] + '\'.')
    
    return
    
# checks for any errors in the configuration
def check_data():
    global model_dir, model_name, image_dir
    
    # check if the given model exists
    if os.path.isdir(model_dir) == False:
        abort('Model directory \'' + str(model_dir) + '\' not found. Try a different value for \'-model_dir\'')
    
    if os.path.isfile(os.path.join(model_dir, model_name)) == False:
        abort('Model not found in \'' + os.path.join(model_dir, model_name) + '\'. Try training a model first by running \'train.py\'')
        
    if os.path.isfile(os.path.join(model_dir, model_name.replace('.hdf5', '.txt'))) == False:
        abort('No categories found for model \'' + str(model_name) + '\'')
        
    # Check if the data directory exists
    if os.path.isdir(image_dir) == False:
        abort('Data directory \'' + str(image_dir) + '\' not found. Try a different value for \'-image_dir\'')
        
    return
    
# processes the images using the ImageProcessor class
def process_data():
    global model_dir, model_name, image_dir, threshold, copy
    
    from lib.image_processor import ImageProcessor
    processor = ImageProcessor()
    processor.model_dir = model_dir
    processor.model_name = model_name
    processor.image_dir = image_dir
    processor.threshold = threshold
    processor.copy = copy
    processor.initialize()
    
# main function
if __name__ == "__main__":
    args = getopts(argv)
    
    if '-help' in args or '-h' in args:
        print_help()
        exit()
        
    parse_args(args)
    check_data()
    
    process_data()
