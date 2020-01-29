import torch
import torch.nn as nn
import torch.nn.functional as F

#seed
torch.manual_seed(7)

class DQN(nn.Module):

    def __init__(self, h,w,outputs):
        super(DQN,self).__init__()
        #Kernalsize = 3, Padding = 1
        self.conv1 = nn.Conv2d(3,3,3,padding=1)
        self.conv2 = nn.Conv2d(3,3,3,padding=1)
        self.fc1 = nn.Linear(12*12*3,256)
        self.fc2 = nn.Linear(256,outputs)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,12*12*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
