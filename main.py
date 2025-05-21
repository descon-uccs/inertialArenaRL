# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:47:12 2025

@author: phili
"""

from classes import InertialContinuousArena
from util import uglyVideo
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

if __name__=="__main__" :
        
    # uglyVideo(10,env)
    
    
    # vec_env = make_vec_env(InertialContinuousArena, n_envs=1, env_kwargs=dict(arena_size=10))
    
    env = InertialContinuousArena()
    
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')
    
    # uglyVideo(10,10,env,model)
    
    # Train the agent
    model.learn(total_timesteps=100_000)
    
    
    # uglyVideo(10,10,env,model)
    
    # mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=1)

    # print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # vec_env.reset()
    # model = A2C("MlpPolicy", vec_env, verbose=1, device='cpu').learn(5000)