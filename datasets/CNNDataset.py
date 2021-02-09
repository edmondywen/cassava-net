import torch
import pandas as pd 
import sklearn
from PIL import Image

#code seems to be stuck in an infinite loop of some sort? isn't printing out anything
class CNNDataset(torch.utils.data.Dataset):

    # 800 x 600 - you want to reduce the image resolution using pytorch (transform) or pillow
    # need some way to load image into a tensor 
    # imagePath should be a string containing file path to where the data is. 
    # use pillow to get a numpy array image.open

    def __init__(self, imagePath = "data/train.csv", trainProp = 0.8, isTrain = True): #size is #images you want to take
        df = pd.read_csv(imagePath) 
        df.sample(0) # shuffle w/ seed. each time we take first 80% last 20% so we should randomize beforehand
        numImages = df.shape[0]

        #sklearn.model_selection.train_test_split(df, test_size = testProp, train_size = trainProp, shuffle = True) prob not needed

        if (isTrain): #note: can remove colon - python will fill in start or end index. not in this case though?
            self.inputImages = df.iloc[0 : int(numImages * trainProp)]
            self.inputImages = self.inputImages['image_id'] #get the entire first column.
            self.truthLabels = df.iloc[0 : int(numImages * trainProp)] 
            self.truthLabels = self.truthLabels['label'] #get the second column
        else: 
            self.inputImages = df.iloc[int(numImages * trainProp): numImages]
            self.inputImages = self.inputImages['image_id']
            self.truthLabels = df.iloc[int(numImages * trainProp): numImages]
            self.truthLabels = self.truthLabels['label']

    def __getitem__(self, index): #allows you to access with square brackets
        #inputs = torch.zeros([3, 224, 224])
        #cache images potentially? 
        input = Image.open("data/train_images/" + self.inputImages[index]) #remove hard code directory
        input = input.resize((224, 224)) #use center crop after if you want the whole image. consider for future
        label = self.truthLabels[index]
        return (input, label)

    def __len__(self): #redefines length
        return len(self.inputImages)