'''
All helper functions are defined here
'''

# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import utilities

def imshow(image, ax=None, title=None):
    '''
    Display the actual image pre-processing
    '''
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, class_to_idx, top_k, device):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    model.to(device)
    model.eval() #Set model in evaluation mode

    # Processing the image
    image_torch1 = utilities.process_image(image_path)
    image_torch2 = torch.from_numpy(image_torch1).type(torch.FloatTensor)
    image_torch3 = image_torch2.unsqueeze_(0)

    with torch.no_grad():
        logpt = model.forward(image_torch3)
        pt = torch.exp(logpt)
        top_p, top_class = pt.topk(top_k, dim=1)

    # Inverting index-class dictionary
    index_to_class = {x: y for y,x in class_to_idx.items()}

    # Converting probabilities and outputs to lists
    top_p_list = np.array(top_p)[0]
    top_index_list = np.array(top_class[0])

    # Converting index list to class list
    top_classes_list = []
    for index in top_index_list:
        top_classes_list += [index_to_class[index]]

    return top_p_list, top_classes_list

def recon(image, probs, classes, cat_to_name):
    '''
    Display the actual image name and the top 5 predicted image names to see the accuracy of
    the predictions on a random image
    '''
    fig, (ax1, ax2) = plt.subplots(figsize=(15, 15), ncols=2)
    ax1.imshow(image)
    ax1.axis('off')

    ax2.barh(np.arange(5), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))

    names = [cat_to_name[c] for c in classes]
    ax1.set_title(names[0])

    ax2.set_yticklabels(names, size='large')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    ax2.invert_yaxis()

    plt.tight_layout()
