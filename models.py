## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch 
        #normalization) to avoid overfitting
        self.num_classes = 136
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        
        # max pooling layers
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # dropout layer
        self.dropout1 = nn.Dropout(0.5)
        
        # Fully connected layer
        self.fc1 = nn.Linear(737280, 512)

        # linear layer (,n_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Input 224 X 224 X 1
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.bn1(x)
        
        #110 X 110 X 32
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.bn2(x)
        
        #53 X 53 X 64
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.bn3(x)
        
        #25 X 25 X 128
        # flatten image input
        x = x.view(-1, 737280) 
        
        # add second hidden layer
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout1(x)

        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
