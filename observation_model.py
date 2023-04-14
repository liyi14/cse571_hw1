import torch.nn as nn

class ObservationModel(nn.Module):
    def __init__(self, output_channels):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2*8*256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,output_channels),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        b,c,h,w = x.shape
        x = x.reshape(b,-1)
        x = self.mlp(x)
        return x
