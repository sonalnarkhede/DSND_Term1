'''
Score the deep learning model already created using train.py to identify different classes of flowers for a given
image or a set of images

Predict flower name from an image with predict.py along with the probability of that name.
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint --gpu
Options:
Use GPU for inference: python predict.py input checkpoint --gpu
Return top KK most likely classes: python predict.py input checkpoint --top_k 3 --gpu
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json --gpu

'''

# Imports here
import argparse
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np

import utilities
import functions
import helper

def main():
    in_args = get_input_args()
    image_path = in_args.input
    checkpoint = in_args.checkpoint
    category_names = in_args.category_names
    top_k = in_args.top_k
    device = 'gpu' if in_args.gpu == 'gpu' else 'cpu'

    #Load checkpoint
    model, input_size, hidden_layers, output_size, epochs, state_dict, optimizer_state, checkpoint, class_to_idx = functions.load_checkpoint(checkpoint)
    print("Loaded model successfully!")

    # Predict input image
    probs, classes = helper.predict(image_path, model, class_to_idx, top_k, device)

    cat_to_name = functions.map_labels(category_names)
    #names = [cat_to_name[c] for c in classes]

    print("Outputs:")
    print(probs, classes)
    #helper.recon(image_path, probs, classes, cat_to_name)


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    5 command line arguements are created:
       input - Path to the image file to predict
       checkpoint - Path to checkpoint file
       category_names - Path to classes map file
       top_k - number of highest probabilities
       gpu - whether to utilize gpu to train
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to the image file to predict')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--category_names', type=str,
                        default='cat_to_name.json', help='Path to classes map file')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of highest probabilities')
    parser.add_argument('--gpu', action='store_true',
                        help='Utilize gpu to train')

    return parser.parse_args()

# Call to main function to run the program
if __name__ == "__main__":
    main()
