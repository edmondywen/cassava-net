import torch
import pandas as pd 
from PIL import Image

class CNNDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    #800 x 600 - you want to reduce the image resolution using pytorch (transform) or pillow
    #need some way to load image into a tensor 
    #imagePath should be a string containing file path to where the data is. use pillow to get a numpy array image.open
    def __init__(self, inputs, labels, imagePath): #try passing in csv
        df = pd.read_csv(imagePath) 
        self.inputImages = df['image_id'] #get the entire 
        self.truthLabels = labels 
        self.imageSource = imagePath


    def __getitem__(self, index): #allows you to access with square brackets
        #inputs = torch.zeros([3, 224, 224])
        inputs = self.inputData[index]
        labels = self.truthLabels[index]
        return (inputs, labels)

    def __len__(self): #redefines length
        return len(self.inputData)

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        pass

    def __getitem__(self, index):
        inputs = torch.zeros([3, 224, 224])
        label = 0

        return inputs, label

    def __len__(self):
        return 10000