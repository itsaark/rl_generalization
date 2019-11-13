__author__ = "Aark Koduru"
__version__ = "0.1"
__email__ = "ark2@pdx.edu"

import gym
import gym_tool_use
import numpy as np
import random
import tensorflow as tf
import tqdm
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from collections import deque, namedtuple

#seed
np.random.seed(7)
random.seed(7)
tf.random.set_seed(7)

#hyperparameters
lr = 0
batch_size = 0
gamma = 0
epsilon_start = 1
epsilon_end = 0.001
e_decay = 0.99
target_policy_update = 0
memory_size = 10_000
episodes = 0

#global variables
SARS = namedtuple('Experience', ('state','action','reward','next_state'))
env = gym.make("TrapTube-v0")
observation = env.reset()
action = env.action_space.sample()
observation_next, reward, done, info = env.step(action)
image = env.render(mode="rgb_array")  # also supports mode="human"


#Intrinsic curiosity module
"""
This module rewards the agent when a new state is reached because of it's action.
"""

#Action value function
class DQN():
    def __init__(self,height,width):
        self.height = height
        self.width = width

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(16,(8,8),strides=4,activation='relu'))
        model.add(layers.Conv2D(32,(4,4),strides=2,activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.Dense(8, activation='linear'))
        model.compile(optimer=RMSprop(learning_rate=lr), loss='Huber', metrics=['accuracy'])
        return model

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
        return random.sample(self.memory,batch_size)

class TrapTubeEnv():
    def __init__(self):
        self.env =  gym.make("TrapTube-v0")
        self.env.reset()
        self.curr_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.curr_screen = None

    def close(self):
        self.env.close()

    def render(self,mode='rgb_array'):
        return self.env.render(mode)

    def actions_count(self):
        return self.env.action_space.n

    def take_action(self,action):
        pass

    def is_starting_state(self):
        return self.curr_screen is None




policy_net = DQN(image.shape[0],image.shape[1]).init_model()
target_net = DQN(image.shape[0],image.shape[1]).init_model()
target_net.set_weights(policy_net.get_weights())

#Training
for episode in tqdm(range(1,episodes+1),unit ='episode'):
    done = False
    observation = env.reset()

    while not done:
        #with probablity epsilon select a random action
        if random.random() < epsilon_start:
            action = env.action_space.sample()
        else:
            #action = np.argmax(policy_net.predict(np.array()))
            pass


if __name__ == "__main__":
    pass
