'''
Model training, validation and scoring related functions defined in this code
'''

# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json

from PIL import Image
import numpy as np

def map_labels(file_path):
    '''
    Function to map flower categories to names
    '''
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def pretrained_model(arch):
    '''
    Function to check which pretrained network has been chosen to build the model and load the corresponding model
    '''
    if arch == 'VGG':
        pretrained_model = models.vgg16(pretrained = True)

    elif arch == 'DenseNet':
        pretrained_model = models.densenet169(pretrained = True)

    else:
        raise Exception("Code only supports VGG and DenseNet. This architecture is not supported!")

    return pretrained_model

def build_model(arch, input_size, hidden_layers, output_size, dropout_rate, device):
    '''
    Define model classifier which would use pretrained network parameters to train the model
    '''
    #Loading a pretrained model
    model = pretrained_model(arch)

    #Freeze parameters of the pre-trained model as we do not want to change model weights and bias
    for param in model.parameters():
        param.requires_grad = False

    #Define classifier parameters
    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_layers),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_rate),
                                     nn.Linear(hidden_layers, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_rate),
                                     nn.Linear(1024, output_size),
                                     nn.LogSoftmax(dim = 1))

    if torch.cuda.is_available() and device == "gpu":
        model.cuda()

    return model

def train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    '''
    Train the model with pretrained network parameters and the custom defined classifier
    '''
    #Turn on dropout
    model.train()

    #Defining the device
    device = torch.device("cuda:0" if device == 'gpu' else "cpu")
    model.to(device)

    #Train the model
    steps = 0

    for epoch in range(epochs):
        running_loss = 0

        for inputs, labels in trainloader:
            steps += 1

            #Move inputs and labels to GPU / CPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() #Initialize optimizer to zero gradient for every training batch

            model.to(device)
            logps = model.forward(inputs) #calculate log probability
            loss = criterion(logps, labels) #calculate the loss
            loss.backward() #backpropagate the loss in the network
            optimizer.step() #optimize the weights and biases

            running_loss += loss.item()

                #Test the model performance on validation set
            if steps % print_every ==0:
                valid_loss, accuracy = validation(model, validloader, criterion)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}.. "
                     )

                running_loss = 0
                #model.train()

def validation(model, validloader, criterion):
    '''
    Test the model built on the validation dataset to calculate and return validation loss and accuracy
    '''
    valid_loss = 0
    accuracy = 0

    #Defining the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Turn off dropout
    model.eval()

    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logpv = model.forward(inputs)
            batch_loss = criterion(logpv, labels)

            valid_loss += batch_loss.item()

            #Calculate accuracy
            pv = torch.exp(logpv)
            top_p, top_class = pv.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #equals = (labels.data == pv.max(dim=1)[1])
            #accuracy += equals.type(torch.FloatTensor).mean()

        model.train()

        return valid_loss, accuracy

def test_accuracy(model, testloader, criterion):
    '''
    Test the model built on the test dataset to calculate the test loss and accuracy
    '''
    test_loss = 0
    accuracy = 0

    #Defining the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Turn off dropout
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logpt = model.forward(inputs)
            batch_loss = criterion(logpt, labels)

            test_loss += batch_loss.item()

            #Calculate accuracy
            pt = torch.exp(logpt)
            top_p, top_class = pt.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(
          f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}.. "
         )

def load_checkpoint(filepath):
    '''
    A function that loads a checkpoint and rebuilds the model
    '''
    # Load the checkpoint
    checkpoint = torch.load(filepath)

    # Load the pre-trained model and rebuild our model
    model = pretrained_model(checkpoint['architecture'])
    input_size = checkpoint['input_size']
    hidden_layers = checkpoint['hidden_layers']
    output_size = checkpoint['output_size']
    epochs = checkpoint['epochs']
    state_dict = checkpoint['state_dict']
    optimizer_state = checkpoint['optimizer_state']
    model.classifier = checkpoint['classifier']
    class_to_idx = checkpoint['class_to_idx']

    print('Checkpoint loaded successfully!')

    return model, input_size, hidden_layers, output_size, epochs, state_dict, optimizer_state, checkpoint, class_to_idx
