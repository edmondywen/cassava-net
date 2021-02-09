import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 4) #4,4 convolution on a 3-channeled image
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(220 * 220 * 3, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x) 
        return x

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(224 * 224 * 3, 1)
        # self.fc2 = nn.Linear(100, 1)
        # self.sigmoid = nn.Sigmoid()
        self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.resnet = self.resnet[:-1] #cut off last layer. fix syntax
        self.resnet.eval()

    def forward(self, x):
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fx2(x)
        # x = self.sigmoid(x)

        with torch.no_grad():
            x = self.resnet(x)

        #do reshaping to make sure output of resnet will fit in 

        #append fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

'''
transfer learning

--don't have a lot of data
--there's already a model out there that has been trained on some data previously 
    -resnet trained on imagenet. classify between trucks, cars, birds, etc
--what we want is to classify cassava leaf disease types:
'''