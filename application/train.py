import sys
from sys import argv
import os

model_dir = 'saved_models'
model_name = 'cnn_image_model.hdf5'
data_dir = 'images'
clear = True
augmentations = 25
patience = 20
test = 0.2
val = 0.2
epochs = 500

def print_help():
    print("------------- HOW TO USE -------------")
    print("")
    print("Purpose:")
    print("This script trains a Convolutional Neural Network to classify images. After the model has been trained, it will be saved as an hdf5 file.")
    print("")
    print("ARGUMENT\tDESCRIPTION\t\t\t\tDEFAULT VALUE")
    print('-model_dir\tDirectory to save the model\t\tsaved_models')
    print('-model_name\tThe name of the finished model\t\tcnn_image_model.hdf5 (will append .hdf5 automatically)')
    print('-data_dir\tRoot directory of the image dataset\timages')
    print('-clear\t\tIf previous datasets should be deleted\tTrue')
    print('-augmentations\tHow augmentations per image\t\t25')
    print('-patience\tHow many epochs before early stopping\t20')
    print('-epochs\tHow many epochs to train the model on\t500')

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
    
def parse_args(args):
    global model_dir, model_name, data_dir, clear, augmentations, patience, test, val, epochs
    
    if '-model_dir' in args:
        model_dir = args['-model_dir']
    if '-model_name' in args:
        model_name = args['-model_name']
        if model_name.endswith('.hdf5') == False:
            model_name = str(model_name) + '.hdf5'
    if '-data_dir' in args:
        data_dir = args['-data_dir']
    if '-clear' in args:
        if args['-clear'].lower() in ['true', 'yes', '1', 'y', 't']:
            clear = True
        elif args['-clear'].lower() in ['false', 'no', '0', 'n', 'f']:
            clear = False
        else:
            abort('Incorrect value for \'-clear\'. Expected \'True\' or \'False\'. Received: \'' + args['-clear'] + '\'.')
    if '-augmentations' in args:
        if args['-augmentations'].isdigit() and int(args['-augmentations']) >= 0:
            augmentations = int(args['-augmentations'])
        else:
            abort('Incorrect value for \'-augmentations\'. Expected positive integer. Received: \'' + args['-augmentations'] + '\'')
    if '-patience' in args:
        if args['-patience'].isdigit() and int(args['-patience']) >= 0:
            patience = int(args['-patience'])
        else:
            abort('Incorrect value for \'-patience\'. Expected positive integer. Received: \'' + args['-patience'] + '\'')
    if '-epochs' in args:
        if args['-epochs'].isdigit() and int(args['-epochs']) >= 0:
            epochs = int(args['-epochs'])
        else:
            abort('Incorrect value for \'-epochs\'. Expected positive integer. Received: \'' + args['-epochs'] + '\'')
    return

# raw_input returns the empty string for "enter"
def query_yes_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    else:
        return False

def abort(error=None):
    print('APPLICATION ABORTED')
    if error != None:
        print('ERROR:')
        print(error)
    exit()
    
def check_data():
    global model_dir, model_name, data_dir, clear, augmentations, patience, test, val, epochs
    
    # check if the directory of images exists. Abort if it doesn't
    if os.path.isdir(data_dir) == False:
        abort('Data directory \'' + str(data_dir) + '\' not found. Try a different value for \'-data_dir\'')
    
    if os.path.isdir(os.path.join(data_dir, 'original')) == False:
        abort('Original dataset not found. Please store the original dataset in location: \'' + os.path.join(data_dir, 'original') + '\'')
    
    # if the output directory doesn't exist, ask to create one
    if os.path.isdir(model_dir) == False:
        if query_yes_no('Directory \'' + str(model_dir) + '\' doesn\'t exist. Create the directory?'):
            os.makedirs(model_dir)
        else:
            abort()
    return
    
def train_model():
    from lib.modeltrainer import ModelTrainer
    
    trainer = ModelTrainer()
    
    trainer.model_dir = model_dir
    trainer.model_name = model_name
    trainer.root_dir = data_dir
    trainer.clear = clear
    trainer.augmentations = augmentations
    trainer.patience = patience
    trainer.test_size = test
    trainer.val_size = val
    trainer.epochs = epochs
    
    trainer.initialize()
    
if __name__ == "__main__":
    args = getopts(argv)
    
    if '-help' in args or '-h' in args:
        print_help()
        exit()
    
    parse_args(args)
    check_data()
    
    train_model()
