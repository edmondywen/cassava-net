import torch


class CNNDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    #imagePath should be a string containing file path to where the data is
    def __init__(self, inputs, labels, imagePath): #try passing in csv
        self.inputData = inputs
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