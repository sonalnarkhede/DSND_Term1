'''
Defining utility functions that perform image pre-processing, datasets load etc. to
get the data prepared before modelling
'''

# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np


def load_data(data_dir):
	'''
	Transforms the incoming images as per the specifications and returns the tranformed data
	that would go as input to the model
	'''

	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	# Define your transforms for the training, validation, and testing sets
	train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      	transforms.Resize(256),
                                      	transforms.CenterCrop(224),
                                      	transforms.RandomHorizontalFlip(),
                                      	transforms.ToTensor(),
                                      	transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      	])

	valid_transforms = transforms.Compose([transforms.Resize(256),
                                      	transforms.CenterCrop(224),
                                      	transforms.ToTensor(),
                                      	transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      	])

	test_transforms = transforms.Compose([transforms.Resize(256),
                                      	transforms.CenterCrop(224),
                                      	transforms.ToTensor(),
                                      	transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      	])

	# Load the datasets with ImageFolder
	train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
	valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
	test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

	# Using the image datasets and the trainforms, define the dataloaders
	trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
	validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)
	testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)

	return train_data, trainloader, validloader, testloader

def process_image(image):
    '''
	Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)

    im_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
    pil_image = im_transform(im)
    np_image = np.array(pil_image)

    return np_image
