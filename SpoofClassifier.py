import requests
import os
import os.path
import pickle
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from PIL import Image  
import torch.nn as nn
import torch
from utils import download_model
import numpy as np
import collections
from extract import *

model = models.efficientnet_b4(pretrained=True)
model.classifier = nn.Sequential(nn.Linear(1792,896),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(896,448),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(448,112),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(112,2))



class SpoofClassifier(object):
    def __init__(self, models_dir = os.getcwd()+'/models'):
        self.model = models.efficientnet_b4(pretrained=True)
        self.model.classifier = nn.Sequential(nn.Linear(1792,896),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(896,448),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(448,112),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(112,2))
        if not os.path.isdir(os.getcwd()+'/models'):
            os.makedirs(os.getcwd()+'/models')
        if os.path.isfile(os.path.join(models_dir,'model_image.pt')):
            pass
        else:
            download_model(models_dir)
        self.model.load_state_dict(torch.load(os.path.join(models_dir,'model_image.pt'),map_location=torch.device('cpu')))
        self.model.eval()
        print('Model Loaded!')
        self.transform = transforms.Compose([transforms.Resize(size=(224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])


    def get_prediction(self,image):
        number_to_class = {0: 'Ham',1: 'Spam'}
        with torch.no_grad():
            self.model.eval()
            # Model outputs log probabilities
            out = self.model(image)
            ps = F.softmax(out, dim =1)
            return number_to_class[int(torch.argmax(ps).item())]

    def detect(self, img_path):
        image = Image.open(img_path).convert('RGB')
        transformed_image = self.transform(image).unsqueeze(0)
        prediction = self.get_prediction(transformed_image)
        return prediction