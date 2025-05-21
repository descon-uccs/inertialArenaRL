# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:58:55 2025

@author: phili
"""

import matplotlib.pyplot as plt

def uglyVideo(seconds,sample_rate,env,model='random',state=None) :
    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(env.render())
    ax.axis("off")
    
    obs,_ = env.reset(state=state)
    for _ in range(seconds*sample_rate):
        if model == 'random' :
            action = env.action_space.sample()
        else :
            action,_ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        img.set_data(env.render())
        if done :
            break
        plt.pause(1/sample_rate)
    
    plt.ioff()
    plt.show()