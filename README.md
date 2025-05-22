# inertialArena

This repository contains two examples of a custom Gymnasium environment and the training process for it using Stable Baselines 3. 
The main purpose of this is to provide useful code examples alongside some basic implemented utilities such as video recording.
The environment is a continuous "gridworld" with a small target at the center. 
The agent uses costly x-y velocity impulses to intercept the target (in the case of the Trigger environment, the intercept is only rewarded if it happens after the trigger time `TA`).

# Installation
Import the included `rl_env.yaml` file into a new Anaconda environment; this should install all of the necessary packages, along with the latest versions of JupyterLab and Spyder.

# Usage
All of the basic functionality can be tested by activating the above anadonda environment and then running
```
python main.py
```
This will:
- Create a basic `InertialContinuousArena` environment
- Record a video with an untrained random agent,
- Train the agent for 10,000 timesteps and create a second video, and
- Train the agent for another 90,000 timesteps and create a third video.

There may be a slight difference between the untrained agent and the partially-trained agent, but the 100k-trained agent should be extremely effective.
The following GIFs show the untrained agent and the agent trained for 100k timesteps.

![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/7512642f-c76b-4059-970a-d3b0d35a43a6)

![100ksteps-step-0-to-step-500-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/7d188268-2a63-4ee3-81e9-99cbffd90485)



