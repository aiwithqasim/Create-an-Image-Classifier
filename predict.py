# Imports here

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

import json

 

import argparse
parsers = argparse.ArgumentParser()
parsers.add_argument('data_dir', action="store")
parsers.add_argument('save_dir', action="store")
parsers.add_argument('--top_k', dest="top_k", default=5)
parsers.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')

parsers.add_argument('--gpu', action="store_const", dest="device", const="gpu", default='cpu')

options = parsers.parse_args()
data_dir = options.data_dir
save_dir = options.save_dir
top_k = options.top_k
category_names = options.category_names
device = options.device



with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# TODO: Write a function that loads a checkpoint and rebuilds the model
checkpoint = torch.load(save_dir+'.pth')
checkpoint.keys()
model = models.vgg16(pretrained=True)
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 500)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.5)),
    ('fc2', nn.Linear(500,102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['state_dict'])

image_path = data_dir

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    image = Image.open(image_path)
    # TODO: Process a PIL image for use in a PyTorch model
    width, height = image.size
    if width > height:
        image.resize((width*256//height,256))
    else:
        image.resize((256,height*256//width))
    
    image = image.crop(((image.width-224)/2, (image.height-224)/2, (image.width-224)/2+224, (image.height-224)/2+224))
    
    np_image = np.array(image)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def imshow(image, ax=None, title=None):
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

def predict(image_path, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img.unsqueeze_(0) # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    result = model.forward(img)
    #print(result)
    probs = torch.exp(result)
    top_probs, top_labals = probs.topk(5)
    return top_probs, top_labals

# TODO: Display an image along with the top 5 classes
def display_img(image_path, model):
    img = process_image(image_path)
    imshow(img, plt.subplot(2,1,1))
    probs, classes = predict(image_path, model)
    probs = probs.tolist()[0]
    classes = classes.tolist()[0]
    flower = [i for i in classes]
    flower_cat = []
    for x in flower:
        for key, value in checkpoint['class_to_idx'].items():
            if value == x:
                flower_cat.append(key)
    flower_name = [cat_to_name[i] for i in flower_cat]
    ax = plt.subplot(2,1,2)
    ax.barh(flower_name, probs)
    plt.show()
    
predict(image_path, model)
display_img(image_path, model)