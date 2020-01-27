import numpy as np
import random
from collections import namedtuple, deque

from model_automata import NatureConvBodySigmoid

import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from LSUV3D import LSUVinit3D

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 8         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, init_data):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.path = 'utils/init_models_LSUV/'
        self.model_LSUV = 'NatureConvBodySigmoid'
        self.init_data = init_data

        # Q-Network
        self.qnetwork_local = NatureConvBodySigmoid(action_size, 1, seed).to(device)
        self.qnetwork_target = NatureConvBodySigmoid(action_size, 1, seed).to(device)
        ## added OMM/
        preinit_models = os.listdir(self.path)
        preinit_models = [i.split('_local.pt')[0] for i in preinit_models]
        if self.model_LSUV in preinit_models:
            print('loading models already preinit with LSUV')
            self.qnetwork_local.load_state_dict(torch.load(f'{self.path}{self.model_LSUV}_local.pt'))
            self.qnetwork_target.load_state_dict(torch.load(f'{self.path}{self.model_LSUV}_target.pt'))
        else:
            self.qnetwork_local.eval()
            self.qnetwork_target.eval()
            with torch.no_grad():
                self.qnetwork_local = LSUVinit3D(self.qnetwork_local, self.init_data)
                self.qnetwork_target = LSUVinit3D(self.qnetwork_target, self.init_data)
            self.qnetwork_local.train()    
            self.qnetwork_target.train()
            torch.save(self.qnetwork_local.state_dict(), f'{self.path}{self.model_LSUV}_local.pt')
            torch.save(self.qnetwork_target.state_dict(), f'{self.path}{self.model_LSUV}_target.pt')
        self.qnetwork_local.to(device)
        self.qnetwork_target.to(device)
        ## /added OMM
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        Q_expected, Q_targets = '_', '_'
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                Q_expected, Q_targets = self.learn(experiences, GAMMA)
        return Q_expected, Q_targets

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        return action_values
        # Epsilon-greedy action selection
        # if random.random() > eps:
        #     return np.argmax(action_values.cpu().data.numpy())
        # else:
        #     return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # print('learning')
        #print(np.shape(states), np.shape(actions), np.shape(rewards), np.shape(next_states))
        # Get max predicted Q values (for next states) from target model
        Q_targets_TEMP = self.qnetwork_target(next_states).detach()
        # print(Q_targets_TEMP.shape)
        # Instead of taking the max we take the average value of the absolute values
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # THESE THREE LINES ONLY PRODUCE A VALUE FOR EACH next_state. 
        # THEY SHOULD BE APPLIED TO Q_expected. THEN MAX() OVER VARIOUS Q_targets OBTAINED
        # FROM SLIGHT VARIATIONS IN THE next_state
        Q_targets_next = self.qnetwork_target(next_states).detach()#.cpu().numpy()
        Q_targets_next = torch.abs(Q_targets_next).sum(1)/Q_targets_next.shape[1]
        Q_targets_next = torch.unsqueeze(Q_targets_next,-1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        ###print(f'Q_targets = {Q_targets[0].squeeze()}')
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # print(f'Q_expected = {Q_expected}')
        # print(f'Q_targets = {Q_targets}')
        
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss #OMM WARNING we are not doing backprop
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)    
        return Q_expected, Q_targets               

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def action_proba_logits_plus_random(action_proba, random_range=2):
    '''Give some randomness to the action_probas.
    1. First convert them to logits, then add some value between some range (random_range).
    Finally convert back to probas'''
    logits = torch.log(action_proba/(1-action_proba))
    random_values = (torch.rand(logits.shape) * random_range - (random_range/2)).to(device)
    logits_plus_random = logits + random_values
    action_proba_random = torch.sigmoid(logits_plus_random)
    return action_proba_random