import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from tqdm import tqdm
from utils_cellular_automata import *
from dqn_agent_automata import Agent, ReplayBuffer, action_plus_random
import imageio
import os

print('State shape: ', '32*32*32')
print('Number of actions: ', 27+27)

nodules_sorted = np.load('/home/om18/Documents/KCL/Nov_27_19_cellular_automata/nodules_sorted.npz')
nodules_sorted = nodules_sorted.f.arr_0
np.shape(nodules_sorted)

nodules_smaller = [i[16:-16, 16:-16, 16:-16] for idx, i in enumerate(nodules_sorted) if idx<1000]
nodules_smaller = np.expand_dims(nodules_smaller,1)
np.shape(nodules_smaller)

# Train across episodes
n_episodes = 11
timesteps = 60
THRESHOLD = .05
n_actions = 27
folder_img_action = 'figures/actions/'
agent = Agent(state_size=(32,32,32), action_size=n_actions+n_actions, seed=0)
preds, neighs, actives, rewards, ndl_targets = [], [], [], [], []
iteration=0
score = 0

score = 0

for i_episode in tqdm(range(n_episodes),total = n_episodes):
    nodule_idx = np.random.randint(0,len(nodules_smaller) - (timesteps+1)) 
    state = nodules_smaller[nodule_idx] # get random nodule
    grid_active = binary_dilation(state[0]>0)
    actives.append(grid_active)
    preds.append(state)
    done = 0

    for j in tqdm(range(timesteps), total=timesteps, leave=False):

        # agent.act
        action_proba = agent.act(state)
        action = [int(i > .5) for i in action_proba] #OMM it used to be .5 or THRESHOLD
        survive = action_plus_random(action[:n_actions], 0, 10, 4)
        birth = action_plus_random(action[n_actions:], 1, n_actions+1, 10, True)
#         print(f'{j} {survive}----{birth}')

        # env.step
        grid_new = copy(state[0])
        grid_neigh, grid_means = count_neighbors_and_get_means_3D_mask(state[0], grid_active, threshold = THRESHOLD) #OMM it used to be .05
        next_state = survive_and_birth_individual_list_3D_mask(grid_neigh, grid_means, grid_active, state[0], survive, birth, grid_active, grid_new)
        nodule_idx += 1
        nodule_target = nodules_smaller[nodule_idx]
        reward_not_normalized = -np.sum(np.abs(nodule_target - next_state)) # THIS SHOULD HAVE A NEGATIVE SIGN
        reward = reward_not_normalized/(np.sum(nodule_target>0)) #normalize by the mask of the target

        # agent.step
        next_state = np.expand_dims(next_state,0); next_state = np.expand_dims(next_state,0)
        state = np.expand_dims(state,0)
        agent.step(state, action, reward, next_state, done) # maybe put action_proba instead of action

        state = next_state[0]
        score += reward

        grid_active = binary_dilation(state[0]>0)
        preds.append(state)
        neighs.append(grid_neigh)
        actives.append(grid_active)
        rewards.append(reward)
        ndl_targets.append(nodule_target)
        
        if i_episode%10 == 0:
            torch.save(agent.qnetwork_local.state_dict(), 'DQN_CA_v0.pth')
        
        if np.abs(reward_not_normalized) > 500: done=1
            
        
        iteration+=1
        if done:
            break 
    figure_action(action_proba, 27, i_episode, folder_img_action)
figure_make_gif(folder_img_action, 'figures/action.gif')