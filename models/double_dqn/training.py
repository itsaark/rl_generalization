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
    print("GPU is running")
else:
    device = torch.device("cpu")

#hyperparameters
lr = 1e-2
batch_size = 128
batch_gd = 32
gamma = 0.9
beta = 0.2
eta = 0.002
epsilon = 0.99
epsilon_end = 0.0001
weight_decay = 0.01
target_policy_update = 32
memory_size = 20_000
episodes = 150

#global variables
SARS = namedtuple('Experience', ('state','action','reward','next_state','done','inverse_loss', 'forward_loss'))

#Different envs
# "PerceptualTrapTube-v0", "StructuralTrapTube-v0", "SymbolicTrapTube-v0",
#"PerceptualStructuralTrapTube-v0","PerceptualSymbolicTrapTube-v0",
#"StructuralSymbolicTrapTube-v0","PerceptualStructuralSymbolicTrapTube-v0"#

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



#ICM
icm = ICM(3,16).to(device)
forward_loss = nn.MSELoss()
inverse_loss = nn.CrossEntropyLoss()
optimizer = Adam(list(agent.parameters()) + list(icm.parameters()),lr=lr,weight_decay=weight_decay)

def run_test(model,test_env,episodes):
    rewards = []
    env = gym.make(test_env)
    for i in range(episodes):
        done = False
        observation = env.reset()
        observation = torch.from_numpy(observation).to(device)
        observation = observation.permute((2, 0, 1))
        observation = observation.unsqueeze(0)
        t_reward = 0
        while not done:
            with torch.no_grad():
                action = Actions[torch.argmax(model(observation))]
            observation_next, reward, done, info = env.step(action)
            #modes = "human","rgb_array"
            image = env.render(mode="rgb_array")
            observation_next = torch.from_numpy(observation_next).to(device)
            observation_next = observation_next.permute((2, 0, 1))
            observation_next = observation_next.unsqueeze(0)

            #ICM
            #converting action space to a tensor
            action_t = torch.from_numpy(one_hot_action(action)).type(torch.FloatTensor).to(device)
            next_s_phi_hat,action_hat,next_s_phi = icm(observation, observation_next, action_t)
            f_loss = forward_loss(next_s_phi_hat, next_s_phi)/2
            target_action_i = Actions.index(list(action))
            i_loss = inverse_loss(action_hat,torch.tensor(target_action_i).unsqueeze(0).to(device))
            i_reward = eta * f_loss.detach()
            reward += i_reward
            t_reward += reward
            observation = observation_next
        rewards.append(t_reward)
    return rewards

def optimize_agent():
    if len(r_memory.memory) < batch_size:
        return
    observation, action, reward, observation_next, done, inverse_loss, forward_loss = r_memory.sample(batch_size)
    observations = torch.cat(observation)
    observation_next = torch.cat(observation_next)
    actions = index_action(torch.LongTensor(action).to(device)).to(device)
    rewards = torch.FloatTensor(reward).to(device)
    inverse_loss = torch.FloatTensor(inverse_loss).to(device)
    forward_loss = torch.FloatTensor(forward_loss).to(device)
    done = torch.FloatTensor(done).to(device)
    q_values = agent(observations)
    p_q_values_next = agent(observation_next)
    q_values_next = target(observation_next)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_value_next = q_values_next.gather(1, torch.max(p_q_values_next, 1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = rewards + gamma * q_value_next * (1 - done)
    icm_loss = (1-beta)*inverse_loss + beta*forward_loss
    loss = ((q_values - expected_q_value.data).pow(2).mean()) + icm_loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def test(agent):
    eval_env = gym.make("PerceptualStructuralSymbolicTrapTube-v0")
    done = False
    observation = eval_env.reset()
    observation = torch.from_numpy(observation).to(device)
    observation = observation.permute((2, 0, 1))
    observation = observation.unsqueeze(0)
    while not done:
        with torch.no_grad():
            action = Actions[torch.argmax(agent(observation))]
            observation_next, reward, done, _ = eval_env.step(action)
        if reward == 1:
            print("solved a task during eval")
            break
        image = eval_env.render(mode="rgb_array")
        observation_next = torch.from_numpy(observation_next).to(device)
        observation_next = observation_next.permute((2, 0, 1))
        observation_next = observation_next.unsqueeze(0)
        observation = observation_next
    return reward

#Training
steps = 0

t_loss = []
t_rewards = []
eval_rewards = []
tasks_solved = 0

for episode in tqdm(range(1,episodes+1),unit ='episode'):
    done = False
    observation = env.reset()
    observation = torch.from_numpy(observation).to(device)
    observation = observation.permute((2, 0, 1))
    observation = observation.unsqueeze(0)
    t_reward = 0
    while not done:
        #with probablity epsilon select a random action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = Actions[torch.argmax(agent(observation))]

        observation_next, reward, done, info = env.step(action)
        if reward == 1:
            tasks_solved += 1
            print("solved a task")
        #modes = "human","rgb_array"
        #image = env.render(mode="rgb_array")
        observation_next = torch.from_numpy(observation_next).to(device)
        observation_next = observation_next.permute((2, 0, 1))
        observation_next = observation_next.unsqueeze(0)

        #ICM
        #converting action space to a tensor
        action_t = torch.from_numpy(one_hot_action(action)).type(torch.FloatTensor).to(device)
        next_s_phi_hat,action_hat,next_s_phi = icm(observation, observation_next, action_t)
        f_loss = forward_loss(next_s_phi_hat, next_s_phi)/2
        target_action_i = Actions.index(list(action))
        i_loss = inverse_loss(action_hat,torch.tensor(target_action_i).unsqueeze(0).to(device))
        i_reward = eta * f_loss.detach()
        reward += i_reward
        t_reward += reward
        sars = SARS(observation, action, reward, observation_next, done, i_loss, f_loss)
        r_memory.update(sars)
        observation = observation_next
        steps += 1
        if steps % batch_gd == 0 and steps > batch_size :
            l = optimize_agent()
            t_loss.append(l.item())
    if episode % target_policy_update == 0:
        target.load_state_dict(agent.state_dict())

    t_rewards.append(t_reward.item())
    #eval_rewards.append(test(agent))

    #Baseline for num of episodes needed to solve a task
    if t_reward > 1:
        break
    if epsilon > epsilon_end:
        epsilon = np.exp(-(1/episodes)*episode*10)

#Saving to files
print(t_loss[:10])
np.save("loss",t_loss)
np.save("training_rewards",t_rewards)
np.save("eval_rewards",eval_rewards)

#Loss plot
t = plt.figure(1)
plt.plot(range((len(t_loss))),list(t_loss),c="r",label="Training loss")
plt.xlabel("Updates")
plt.ylabel("Loss")
plt.legend()
t.savefig("Training_loss.png")

#Rewards plot
r = plt.figure(2)
plt.plot(range(len(t_rewards)),t_rewards,c="b")
plt.xlabel("Episodes")
plt.ylabel("Reward")
r.savefig("Reward.png")

print(f"Solved {tasks_solved} tasks in {episodes} episodes")

if __name__ == "__main__":
    pass
