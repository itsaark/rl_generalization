import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)


class Features(nn.Module):
    def __init__(self,input_size):
        super(Features,self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(input_size, 3,3,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(3,3,3,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(3,3,3,padding=1),
                    nn.ReLU()
                    )
    def forward(self,x):
        return self.conv(x)

class Forward_Model(nn.Module):
    def __init__(self,action_size):
        super(Forward_Model,self).__init__()
        self.linear = nn.Sequential(
                            nn.Linear((12*12*3)+action_size, 512),
                            nn.LeakyReLU(),
                            nn.Linear(512,12*12*3)
                            )
    def forward(self,x):
        return self.linear(x)

class Inverse_Model(nn.Module):
    def __init__(self,action_size):
        super(Inverse_Model,self).__init__()
        self.linear = nn.Sequential(
                            nn.Linear(12*12*3*2, 512),
                            nn.LeakyReLU(),
                            nn.Linear(512,action_size)
                            )
    def forward(self,x):
        return self.linear(x)

class ICM(nn.Module):

    def __init__(self, input_size,action_size):
        super(ICM,self).__init__()
        self.conv = Features(input_size)
        self.forward_model = Forward_Model(action_size)
        self.inverse_model = Inverse_Model(action_size)

    def forward(self,state,next_state,action):
        state_ft = self.conv(state)
        next_state_ft = self.conv(next_state)
        s_phi = state_ft.view(-1, 12*12*3)
        next_s_phi = state_ft.view(-1, 12*12*3)
        return self.forward_model(torch.cat((s_phi,action),1)), self.inverse_model(torch.cat((s_phi,next_s_phi),1)), next_s_phi
