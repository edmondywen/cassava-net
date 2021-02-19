import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetwork(torch.nn.Module):
    def __init__(self, input_channels, output_dim):
        #super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #stride 2 padding 0 to reduce image size by factor of 2. default padding is 0
        #image is now reduced to 112x112

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #image is now 56x56

        self.fc1 = nn.Linear(56 * 56 * 64, 1000)
        self.fc2 = nn.Linear(1000, 5) 

        '''
        self.fc1 = torch.nn.Linear() #in feature = out of prev
        self.fc2 = torch.nn.Linear()
        self.relu = torch.relu
        self.conv1 = torch.Conv2d(input_channels, 5, kernel_size=4, stride=1, padding=0) #technically a member variable. you can call objects like functions tho. 
        self.conv2 = torch.Conv2d(32, 5, kernel_size=4, stride=1, padding=0) #input_channels usually go from 3 -> 32 -> 64 -> 128 and shrinking image to compensate ie w/ pooling or with increasing stride (skipping over pixels)
        self.pool = torch.maxpool()
        '''

        #conv2d args are inputchannels (3 for rgb), outchannels. trying 5 and let's see what happens
        # self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        # self.resnet = self.resnet[:-1] #cut off last layer. fix syntax
        # self.resnet.eval()
    
    #overrides the base function's forward 
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) #flatten
        out = self.fc1(out)
        out = self.fc2(out)
        return out 
        
        '''
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.pool(x)
        x = self.fc2(x)
        '''

        # transfer learning - comment out above to implement
        # with torch.no_grad():
        #     x = self.resnet(x)

        # #do reshaping to make sure output of resnet will fit in 

        # #append fully connected layers
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)

        return(x)

