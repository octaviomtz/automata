import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

%load_ext autoreload
%autoreload 2

from tqdm import tqdm_notebook
import seaborn as sns
from utils_cellular_automata import *
from dqn_agent_automata import Agent

print('State shape: ', '32*32*32')
print('Number of actions: ', 27+27)

nodules_sorted = np.load('/home/om18/Documents/KCL/Nov_27_19_cellular_automata/nodules_sorted.npz')
nodules_sorted = nodules_sorted.f.arr_0
np.shape(nodules_sorted)

nodules_smaller = [i[16:-16, 16:-16, 16:-16] for idx, i in enumerate(nodules_sorted) if idx<400]
nodules_smaller = np.expand_dims(nodules_smaller,1)
np.shape(nodules_smaller)

def action_plus_random(action, min_rand_action_val=0, max_rand_action_val=10, rand_values = 4, birth_no_0=False):
    '''Convert to integers and add randomness
    1.Get survive/birth from the full action array. 
    2.Add randomness WARNING (check if this is correct) 
    Note1. Prevent nodules growing from nothing: Birth min_rand_action_val should be 1 and birth_no_0 = True'''
    action = np.where(np.asarray(action)==1)[0]
    # Randomness
    rand_action = np.random.randint(min_rand_action_val, max_rand_action_val, rand_values)
    action = np.random.permutation(action)
    # If the rand_values are more than the values in action, then insert zeros that will be replaced by rand_values
    if rand_values > len(action):
        vals_to_insert = list(np.zeros(rand_values - len(action)))
        action = np.asarray(vals_to_insert + list(action))
    np.put(action, list(range(rand_values)),list(rand_action))
    action = np.sort(action)
    action = np.unique(action)
    if birth_no_0:
        action = np.delete(action,0)
    return action

# Train across episodes
n_episodes = 500
timesteps = 60
n_actions = 27
agent = Agent(state_size=(32,32,32), action_size=n_actions+n_actions, seed=0)
preds, neighs, actives, rewards, ndl_targets = [], [], [], [], []

score = 0

for i_episode in tqdm_notebook(range(1, n_episodes+1),total = n_episodes+1):
    nodule_idx = np.random.randint(0,len(nodules_smaller) - (timesteps+1)) 
    state = nodules_smaller[nodule_idx] # get random nodule
    grid_active = binary_dilation(state[0]>0)
    actives.append(grid_active)
    preds.append(state)
    done = 0

    for j in tqdm_notebook(range(timesteps), total=timesteps, leave=False):

        # agent.act
        action_proba = agent.act(state)
        action = [int(i >.5) for i in action_proba]
        survive = action_plus_random(action[:n_actions], 0, 10, 4)
        birth = action_plus_random(action[n_actions:], 1, n_actions+1, 10, True)
#         print(f'{j} {survive}----{birth}')

        # env.step
        grid_new = copy(state[0])
        grid_neigh, grid_means = count_neighbors_and_get_means_3D_mask(state[0], grid_active, threshold=.5)
        next_state = survive_and_birth_individual_list_3D_mask(grid_neigh, grid_means, grid_active, state[0], survive, birth, grid_active, grid_new)
        nodule_idx += 1
        nodule_target = nodules_smaller[nodule_idx]
        reward_not_normalized = -np.sum(np.abs(nodule_target - next_state)) # THIS SHOULD HAVE A NEGATIVE SIGN
        reward = reward_not_normalized/(np.sum(nodule_target>0)) #normalize by the mask of the target

        # agent.step
        next_state = np.expand_dims(next_state,0); next_state = np.expand_dims(next_state,0)
        state = np.expand_dims(state,0)
        agent.step(state, action, reward, next_state, done)

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
            
        if done:
            break 