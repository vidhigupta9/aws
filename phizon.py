import requests
import os
import os.path
import pickle
from sqlalchemy import true
import torch
from utils import download_model
from extract import extract_url
from model import Net
import numpy as np
import collections
from extract import *
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image  
import torch.nn as nn

class UrlDetector(object):
    def __init__(self, models_dir = os.getcwd()+'/models'):
        self.model = Net()
        if os.path.isfile(os.path.join(models_dir,'scaler.pkl')):
            self.scaler = pickle.load(open(os.path.join(models_dir,'scaler.pkl'), 'rb'))
        else:
            download_model(models_dir)
            self.scaler = pickle.load(open(os.path.join(models_dir,'scaler.pkl'), 'rb'))
        if os.path.isfile(os.path.join(models_dir,'model.pkl')):
            self.model.load_state_dict(torch.load(os.path.join(models_dir,'model.pt'),map_location=torch.device('cpu')))
        else:
            download_model(models_dir)
            self.model.load_state_dict(torch.load(os.path.join(models_dir,'model.pt'),map_location=torch.device('cpu')))
        #self.model.load_state_dict(torch.load(os.path.join(models_dir,'model.pt'),map_location=torch.device('cpu')))

    def urldetect(self, url):
        features = extract_url(url)
        features = [w.replace('True', '1').replace('False', '0') for w in np.array(features).astype(str)]
        features = np.array((features)).astype(float)
        features = torch.tensor(self.scaler.transform(features.reshape(1,-1)).ravel())
        output = self.model(features.float().unsqueeze(0))
        _, pred = torch.max(output, 1)
        return pred.item()

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
        with torch.no_grad():
            self.model.eval()
            # Model outputs log probabilities
            out = self.model(image)
            ps = F.softmax(out, dim =1)
            return int(torch.argmax(ps).item())

    def spoofdetect(self, img_path):
        image = Image.open(img_path).convert('RGB')
        transformed_image = self.transform(image).unsqueeze(0)
        prediction = self.get_prediction(transformed_image)
        return prediction