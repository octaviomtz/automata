import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from tqdm import tqdm_notebook
from utils_cellular_automata import *
from dqn_agent_automata import Agent, ReplayBuffer, action_proba_logits_plus_random
import imageio
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
np.set_printoptions(precision=3)

print('State shape: ', '32*32*32')
print('Number of actions: ', 27+27)

nodules_sorted = np.load('/home/om18/Documents/KCL/Nov_27_19_cellular_automata/nodules_sorted.npz')
nodules_sorted = nodules_sorted.f.arr_0
nodules_smaller = [i[16:-16, 16:-16, 16:-16] for idx, i in enumerate(nodules_sorted) if idx<1000]
nodules_smaller = np.expand_dims(nodules_smaller,1)
np.shape(nodules_sorted), np.shape(nodules_smaller)

# Dataloader used to get LSUV init
dataset = dataset_nodules(np.squeeze(nodules_smaller), transform=True)
dataloader = DataLoader(dataset, batch_size=32)
dataiter = iter(dataloader)
init_data = next(dataiter)
init_data.shape

# Train across episodes
n_episodes = 2 # 11
timesteps = 60
THRESHOLD = .5
n_actions = 27
folder_img_action = 'figures/actions/'
agent = Agent(state_size=(32,32,32), action_size=n_actions+n_actions, seed=0, init_data=init_data)
preds, neighs, actives, rewards, ndl_targets = [], [], [], [], []
iteration=0
score = 0
action_probas, action_proba_randoms, action_proba_random_manys = [], [], []

for i_episode in tqdm_notebook(range(n_episodes),total = n_episodes):
    nodule_idx = np.random.randint(0,len(nodules_smaller) - (timesteps+1)) 
    state = nodules_smaller[nodule_idx] # get random nodule
    grid_active = binary_dilation(state[0]>0)
    actives.append(grid_active)
    preds.append(state)
    done = 0

    for j in tqdm_notebook(range(timesteps), total=timesteps, leave=False):
        # nodule_target is used in env.step
        nodule_idx += 1
        nodule_target = nodules_smaller[nodule_idx]
        
        # agent.act
        action_proba = agent.act(state)
        action_proba_random_many = [action_proba_logits_plus_random(action_proba) for i in range(3)]
        action_proba_random = action_proba_random_many[0]
        action = np.round(action_proba_random.detach().cpu().numpy(),0)
        
        # env.step
        next_state, reward, reward_not_normalized, grid_neigh = env_step(state[0], grid_active, nodule_target, action, THRESHOLD)
    
        # agent.step
        state = np.expand_dims(state,0)
        Q_expected, Q_targets = agent.step(state, action, reward, next_state, done) # maybe put action_proba instead of action
    
        print(f'({iteration:02d})Q_targets={Q_targets}')
#         print(f'Q_expected={Q_expected}')
    
        state = next_state[0]
        score += reward

        grid_active = binary_dilation(state[0]>0)
        preds.append(state)
        neighs.append(grid_neigh)
        actives.append(grid_active)
        rewards.append(reward)
        ndl_targets.append(nodule_target)
        #
        action_probas.append(action_proba)
        action_proba_random_manys.append(action_proba_random_many)
        action_proba_randoms.append(action_proba_random)
        
        if i_episode%10 == 0:
            torch.save(agent.qnetwork_local.state_dict(), 'DQN_CA_v0.pth')
        
        if np.abs(reward_not_normalized) > 500: done=1
            
        
        iteration+=1
        if done:
            break 
    figure_action(action_proba.detach().cpu().numpy(), 27, i_episode, folder_img_action)
figure_make_gif(folder_img_action, 'figures/action.gif')