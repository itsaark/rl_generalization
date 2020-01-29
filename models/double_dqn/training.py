import gym
import gym_tool_use
import numpy as np
import random
from tqdm import tqdm
from collections import deque, namedtuple
from dqn_agent import DQN
import torch
from torch.optim import Adam
#seed
np.random.seed(7)
random.seed(7)

#hyperparameters
lr = 0
batch_size = 10
gamma = 0.5
epsilon_start = 0.8
epsilon_end = 0.001
e_decay = 0.99
target_policy_update = 5
memory_size = 10_000
episodes = 10

#global variables
SARS = namedtuple('Experience', ('state','action','reward','next_state','done'))
env = gym.make("TrapTube-v0")

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
agent = DQN(12,12,16)
target = DQN(12,12,16)
target.load_state_dict(agent.state_dict())
optimizer = Adam(agent.parameters())

def update_target():
    if len(r_memory.memory) < batch_size:
        return
    observation, action, reward, observation_next, done = r_memory.sample(batch_size)
    observations = torch.cat(observation)
    observation_next = torch.cat(observation_next)
    actions = index_action(torch.LongTensor(action))
    rewards = torch.LongTensor(reward)
    done = torch.FloatTensor(done)
    q_values = agent(observations)
    p_q_values_next = agent(observation_next)
    q_values_next = target(observation_next)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_value_next = q_values_next.gather(1, torch.max(p_q_values_next, 1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = rewards + gamma * q_value_next * (1 - done)
    loss = (q_values - expected_q_value.data).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

#Training
steps = 0

for episode in tqdm(range(1,episodes+1),unit ='episode'):
    done = False
    observation = env.reset()
    observation = torch.from_numpy(observation)
    observation = observation.permute((2, 0, 1))
    observation = observation.unsqueeze(0)
    while not done:
        #print(observation.unsqueeze(0).shape)
        #with probablity epsilon select a random action
        if random.random() < epsilon_start:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = Actions[np.argmax(agent(observation))]

        observation_next, reward, done, info = env.step(action)
        image = env.render(mode="human")
        observation_next = torch.from_numpy(observation_next)
        observation_next = observation_next.permute((2, 0, 1))
        observation_next = observation_next.unsqueeze(0)
        #ICM
        #i_reward =
        #reward += i_reward
        sars = SARS(observation, action, reward, observation_next, done)
        r_memory.update(sars)
        observation = observation_next
        steps += 1
        if steps % target_policy_update == 0:
            update_target()

if __name__ == "__main__":
    pass
