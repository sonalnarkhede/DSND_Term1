'''
Train a deep learning network to identify different classes of flowers for a given image or a set of images

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
'''

# Imports here
import argparse
import os

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np

import utilities
import functions
import helper

def main():
    in_args = get_input_args()
    print('Received inputs from the user is: ', in_args)

    data_dir = in_args.data_dir
    arch = in_args.arch #supports only vgg and densenet
    save_dir = in_args.save_dir
    device = 'gpu' if in_args.gpu == 'gpu' else 'cpu'

    if arch not in ['VGG', 'DenseNet']:
        raise Exception("Code only supports VGG and DenseNet. This architecture is not supported!")

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Load datasets
    train_data, trainloader, validloader, testloader = utilities.load_data(data_dir)
    print("Loaded the datasets!")
    
    #Load the pretrained model
    pretrained_model = functions.pretrained_model(arch)
    print("Loaded the pretrained model!")
    
    # Get pretrained model name
    model_name = pretrained_model.__class__.__name__

    # Get hyperparameters from command line arguments
    if (model_name == "DenseNet"):
        input_size = pretrained_model.classifier.in_features
    elif (model_name == "VGG"):
        input_size = pretrained_model.classifier[0].in_features
    else:
        raise Exception("Code only supports VGG and DenseNet. This architecture is not supported!")

    hidden_layers = in_args.hidden_units
    output_size = 102
    epochs = in_args.epochs
    learning_rate = in_args.learning_rate
    dropout_rate = 0.2

    #Build the model
    model = functions.build_model(arch, input_size, hidden_layers, output_size, dropout_rate, device)
    print("Built the model architecture!")
    
    #Define the loss criterion
    criterion = nn.NLLLoss()

    #Define optimizer for classifier parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #Train the model
    print_every = 5
    print("Model training started!")
    functions.train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device)
    print("Model training completed!")
    
    print("Model testing started!")
    functions.test_accuracy(model, testloader, criterion)
    print("Model testing completed!")

    # Save the checkpoint
    checkpoint = {'architecture': model_name,
              'model': model,
              'input_size': input_size,
              'hidden_layers': hidden_layers,
              'output_size': output_size,
              'learning_rate': learning_rate,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': train_data.class_to_idx
                 }
    # Create a folder to save checkpoint if not already existed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # file name format: {architecture}_{input_size}_{hidden_layers}_{output_size}_{epochs}_{learning_rate}_checkpoint.pth
    torch.save(checkpoint, save_dir + '/{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(checkpoint['architecture'],
          checkpoint['input_size'], checkpoint['hidden_layers'], checkpoint['output_size'],
          checkpoint['epochs'], checkpoint['learning_rate']))
    #torch.save(checkpoint, 'checkpoint.pth')

    print('Model checkpoint saved successfully!')

def get_input_args():
    '''
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    7 command line arguements are created:
       data_dir - Path to the image files
       arch - pretrained CNN model architecture to use for image classification (default-
              pick any of the following vgg, densenet)
       save_dir - Set directory to save checkpoints
       learning_rate - learning rate for optimizer
       hidden_units - number of hidden units
       epochs - number of epochs
       gpu - whether to utilize gpu to train
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to image files')
    parser.add_argument('--arch', type=str, default='VGG', help='CNN model architecture to use for image classification')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate for optimizer')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Utilize gpu to train')

    return parser.parse_args()

# Call to main function to run the program
if __name__ == "__main__":
    main()
