# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:51:09 2020

@author: Aamod Save
"""

from unityagents import UnityEnvironment
from ddpg_agent import DDPGAgent
from torch.optim import Adam

#Number of test episodes
n_test = 5
t_max = 1000

#Initialize hyperparameters
lr_actor = 1.5e-4
lr_critic = 1.5e-4
weight_decay = 0.0001
tau = 1e-3
gamma = 0.99
seed = 3

#Make a kwargs dictionary
kwargs = {'actor_optim': Adam, 'critic_optim': Adam, 'lr_actor': lr_actor,
          'lr_critic': lr_critic, 'tau': tau, 'seed': seed, 'weight_decay': weight_decay}

#Path to environment
path_to_env = 'Reacher_Windows_x86_64/Reacher.exe'

#Path to saved model
model_path = "trained_models/"

#Main function
if __name__ == '__main__':
    
    #Load the environment
    env = UnityEnvironment(path_to_env)
    
    #Get the banana brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    #Reset the env to get state and action space
    env_info = env.reset()[brain_name]

    #Get the state an actions spaces
    state_space = len(env_info.vector_observations[0])
    action_space = brain.vector_action_space_size
    
    #Create the agent
    agent = DDPGAgent(state_space, action_space, **kwargs)
    
    #Load the saved model
    agent.load(model_path)
    
    #Watch the smart agent play
    for n in range(n_test):
        
        rewards = 0
        
        #Reset the env
        env_info = env.reset()[brain_name]
        state = env_info.vector_observations[0]
        
        done = False
        
        t = 0
        
        while t <= t_max:
        
            #Choose an action
            action = agent.act(state)
            
            #Perform the action
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            state = next_state
            rewards += reward
            
            if done or t == t_max:
                print("Episode finished in {} timesteps!".format(t))
                print("Rewards earned: {}".format(rewards))
                break
            
            t += 1
            
    #Close the environment
    env.close()