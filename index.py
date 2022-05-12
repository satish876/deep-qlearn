import gym
import gym.spaces
from gym import envs
import time
from matplotlib import pyplot as PLT
import numpy as np
from PIL import Image

DEFAULT_ENV_NAME ="AirRaid-v4"
env = gym.make(DEFAULT_ENV_NAME, render_mode="rgb_array")
env.reset()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


rew_tot = 0
for _ in range(5):
    action = env.action_space.sample() #take step using random action from possible actions (actio_space)
    obs, rew, done, info = env.step(action) 
    rew_tot = rew_tot + rew
    # print("Reward: %r" % rew_tot)  
    # print(obs.shape, obs[...,:3].shape)
    # PLT.imshow(obs)
    # PLT.show()
    gray = rgb2gray(obs)    
    PLT.imshow(gray)
    # # PLT.imshow(gray, cmap=PLT.get_cmap('gray'), vmin=0, vmax=1)
    PLT.show()

    # time.sleep(5)
    # env.render()
#Print the reward of these random action


# img = mpimg.imread('image.png')     
