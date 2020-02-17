import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)

class ICM(nn.Module):

    def __init__(self, input_size,action_size):
        super(ICM,self).__init__():
        self.conv = nn.Sequential(
                    nn.Conv2d(input_size, 3,3,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(3,3,3,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(3,3,3,padding=1),
                    nn.ReLU()
                    )
        self.forward_model = nn.Sequential(
                            nn.Linear(12*12*3+action_size, 512),
                            nn.LeakyReLU(),
                            nn.Linear(512,12*12*3)
                            )
        self.inverse_model = nn.Sequential(
                            nn.Linear(12*12*3*2, 512),
                            nn.LeakyReLU(),
                            nn.Linear(512,action_size)
                            )


    def forward(self,state,next_state,action):
        s_phi = self.conv(state)
        next_s_phi = self.conv(next_state)
        return self.forward_model(s_phi,action),self.inverse_model(torch.cat(s_phi,next_s_phi)),next_s_phi
