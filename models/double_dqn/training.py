import gym
import gym_tool_use
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from collections import deque, namedtuple
from ddqn_agent import DQN
from ICModule import ICM

#seed
np.random.seed(7)
random.seed(7)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

#hyperparameters
lr = 0
batch_size = 10
gamma = 0.5
beta = 0.1
eta = 0.1
epsilon_start = 0.8
epsilon_end = 0.001
e_decay = 0.99
target_policy_update = 5
memory_size = 10_000
episodes = 10

#global variables
SARS = namedtuple('Experience', ('state','action','reward','next_state','done'))
env = gym.make("StructuralTrapTube-v0")

North = 0
South = 1
West = 2
East = 3

Actions = [[North,North],
            [North, South],
            [North, East],
            [North, West],
            [South, North],
            [South, South],
            [South, East],
            [South, West],
            [East, North],
            [East, South],
            [East, East],
            [East, West],
            [West, North],
            [West, South],
            [West, East],
            [West, West]]

def index_action(a):
    a = a.tolist()
    result = []
    for action in a:
        result.append(Actions.index(action))
    return torch.LongTensor(result)

def one_hot_action(a):
    result = np.zeros((1,16))
    result[0,a] = 1
    return result

class ReplayMemory():
    """
    This is method returns a tuple which contains the state which the agent has been
    to before, the action it took, the reward it got and the new state it reached
    at that time
    """
    def __init__(self,size):
        self.size = size
        self.memory = deque(maxlen=self.size)

    def update(self,SARS):
        self.memory.append(SARS)

    def sample(self,batch_size):
        return zip(*random.sample(self.memory,batch_size))


r_memory = ReplayMemory(memory_size)
agent = DQN(12,12,16).to(device)
target = DQN(12,12,16).to(device)
target.load_state_dict(agent.state_dict())
optimizer = Adam(agent.parameters())

#ICM
icm = ICM(3,16)
forward_loss = nn.MSELoss()
inverse_loss = nn.CrossEntropyLoss()

def update_target(inverse_loss,forward_loss):
    if len(r_memory.memory) < batch_size:
        return
    observation, action, reward, observation_next, done = r_memory.sample(batch_size)
    observations = torch.cat(observation)
    observation_next = torch.cat(observation_next)
    actions = index_action(torch.LongTensor(action).to(device))
    rewards = torch.LongTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)
    q_values = agent(observations)
    p_q_values_next = agent(observation_next)
    q_values_next = target(observation_next)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_value_next = q_values_next.gather(1, torch.max(p_q_values_next, 1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = rewards + gamma * q_value_next * (1 - done)
    icm_loss = (1-beta)*inverse_loss + beta*forward_loss
    loss = ((q_values - expected_q_value.data).pow(2).mean()) + icm_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

#Training
steps = 0

t_loss = []
t_rewards = []

for episode in tqdm(range(1,episodes+1),unit ='episode'):
    done = False
    observation = env.reset()
    observation = torch.from_numpy(observation).to(device)
    observation = observation.permute((2, 0, 1))
    observation = observation.unsqueeze(0)
    while not done:
        #with probablity epsilon select a random action
        if random.random() < epsilon_start:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = Actions[np.argmax(agent(observation))]

        observation_next, reward, done, info = env.step(action)
        #modes = "human","rgb_array"
        image = env.render(mode="human")
        observation_next = torch.from_numpy(observation_next).to(device)
        observation_next = observation_next.permute((2, 0, 1))
        observation_next = observation_next.unsqueeze(0)

        #ICM
        #converting action space to a tensor
        action_t = torch.from_numpy(one_hot_action(action)).type(torch.FloatTensor).to(device)
        next_s_phi_hat,action_hat,next_s_phi = icm(observation, observation_next, action_t)
        f_loss = forward_loss(next_s_phi_hat, next_s_phi)/2
        target_action_i = Actions.index(list(action))
        i_loss = inverse_loss(action_hat,torch.tensor(target_action_i).unsqueeze(0))
        i_reward = eta * f_loss.detach()
        reward += i_reward

        sars = SARS(observation, action, reward, observation_next, done)
        r_memory.update(sars)
        observation = observation_next
        steps += 1
        if steps % target_policy_update == 0:
            l = update_target(i_loss,f_loss)
            t_loss.append(l)

#Plotting loss
plt.plot(len(t_loss),list(t_loss),c="c3",label="Training loss")
plt.savefig("Training_loss.png")


if __name__ == "__main__":
    pass
