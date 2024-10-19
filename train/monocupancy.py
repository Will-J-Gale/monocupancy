from torch import nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.ops import Conv2dNormActivation

class Monocupancy(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()

        self.imageInput = Conv2dNormActivation(in_channels, 32, kernel_size=3, stride=2, activation_layer=nn.SiLU)
        self.efficientnet = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        self.features = nn.Sequential(
            self.efficientnet.features[1:],
        )

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1408, 1408, kernel_size=[4,2])
        self.convt1 = nn.ConvTranspose2d(1408, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convt3 = nn.ConvTranspose2d(256, 120, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1x1 = nn.Conv2d(120, 120, kernel_size=1)

    def forward(self, imageInputs):
        #Extract image features
        x = self.imageInput(imageInputs)
        x = self.features(x)

        #Decode to occupancy grid 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.convt1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.convt2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.convt3(x)
        x = self.relu(x)
        x = self.conv1x1(x)

        return x