import torch
import pandas as pd 
from PIL import Image

class CNNDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    # 800 x 600 - you want to reduce the image resolution using pytorch (transform) or pillow
    # need some way to load image into a tensor 
    # imagePath should be a string containing file path to where the data is. 
    # use pillow to get a numpy array image.open

    def __init__(self, inputs, labels, imagePath): #try passing in csv
        df = pd.read_csv(imagePath) 
        self.inputImages = df['image_id'] #get the entire first column
        self.truthLabels = labels['label'] #get the second column

    def __getitem__(self, index): #allows you to access with square brackets
        #inputs = torch.zeros([3, 224, 224])
        #cache images potentially? 
        input = Image.open(self.inputImages[index])
        input = input.resize((224, 224)) #use center crop after if you want the whole image 
        label = self.truthLabels[index]
        return (input, label)

    def __len__(self): #redefines length
        return len(self.inputImages)