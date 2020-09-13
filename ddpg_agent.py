# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:20:48 2020

@author: Aamod Save
"""

from networks import Actor, Critic
import torch
import copy
from torch.nn.functional import mse_loss
import numpy as np
import os
import random

class DDPGAgent:
    
    def __init__(self, state_space, action_space, **kwargs):
        
        self.state_space = state_space
        self.action_space = action_space
        
        self.actor_optim = kwargs['actor_optim']
        self.critic_optim = kwargs['critic_optim']
        self.lr_actor = kwargs['lr_actor']
        self.lr_critic = kwargs['lr_critic']
        self.tau = kwargs['tau']
        self.seed = random.seed(kwargs['seed'])
        self.weight_decay = kwargs['weight_decay']
        
        #Set the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        #Actor network
        self.actor_local = Actor(self.state_space, self.action_space, kwargs['seed']).to(self.device)
        self.actor_target = Actor(self.state_space, self.action_space, kwargs['seed']).to(self.device)
        self.actor_optimizer = self.actor_optim(self.actor_local.parameters(), lr=self.lr_actor)
        
        #Critic network
        self.critic_local = Critic(self.state_space, self.action_space, kwargs['seed']).to(self.device)
        self.critic_target = Critic(self.state_space, self.action_space, kwargs['seed']).to(self.device)
        self.critic_optimizer = self.critic_optim(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)
        
        #Noise
        self.noise = Noise(self.action_space, self.seed)
        
    def train(self, experiences, gamma=1.0):
        
        #Get observations from the memory
        states, actions, rewards, next_states, dones = experiences
        
        #Calculate the loss using temporal difference
        next_actions = self.actor_target(next_states)
        
        #Use these actions to calculate target values for critic
        targets = self.critic_target(next_states, next_actions)
        targets = rewards + (gamma * targets * (1-dones))
        
        #Calculate predictions from local critic
        preds = self.critic_local(states, actions)
        
        #Calculate the critic loss
        critic_loss = mse_loss(preds, targets)
        
        #Update the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #Calculate actor loss
        action_preds = self.actor_local(states)
        actor_loss = -self.critic_local(states, action_preds).mean()
        
        #Update the local actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #Perform soft update
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        
    def act(self, state, add_noise=False):
        
        state = torch.from_numpy(state).float().to(self.device)
        
        #Set the actor to eval mode
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        #Set the actor in training mode
        self.actor_local.train()
        
        #Add noise to the action
        if add_noise:
            action += self.noise.sample()
            
        return np.clip(action, -1, 1)
    
    def soft_update(self, local, target, tau=1e-3):
    
        #Perform a soft update of the network
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*(local_param.data) + (1.0-tau)*(target_param.data))
            
    def save(self, path):
        torch.save(self.actor_local.state_dict(), os.path.join(path, 'actor'))
        torch.save(self.critic_local.state_dict(), os.path.join(path, 'critic'))
        
    def load(self, path):
        self.actor_local.load_state_dict(torch.load(os.path.join(path, 'actor')))
        self.critic_local.load_state_dict(torch.load(os.path.join(path, 'critic')))
        
#Add a class for Noise 
class Noise:
    
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        
        self.mu = mu * np.ones(size)
        self.size = size
        self.seed = random.seed(seed)
        self.theta = theta
        self.sigma=sigma
        self.reset()
        
    def reset(self):
        #Reset the internal state
        self.state = copy.copy(self.mu)
        
    def sample(self):
        #Add noise according to Ornstein-Uhlenbeck process
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
        