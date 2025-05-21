# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:58:55 2025

@author: phili
"""

import matplotlib.pyplot as plt
import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'
import base64
from pathlib import Path
from IPython import display as ipythondisplay
import gymnasium as gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


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
    
def show_videos(video_path="", prefix=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
    
def record_video(env_id, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()