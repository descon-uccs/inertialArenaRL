# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:47:35 2025

@author: phili
"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

class InertialContinuousArena(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an environment where the agent has impulsive control and 
    must learn to enter a 1x1 box at the origin.
    """

    metadata = {"render_modes": ["console",'rgb_array']}

    # Define constants for clearer code
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    COAST = 4

    action_list = [np.array([0,1]),  # UP
                   -np.array([0,1]), # DOWN
                   np.array([1,0]),  # RIGHT
                   -np.array([1,0]), # LEFT
                   np.array([0,0])]  # COAST

    GOAL = spaces.Box(np.array([-1,-1]),
                      np.array([1,1]),
                      shape=(2,),
                      dtype=np.float64) # GOAL is a 1x1 box around the origin. position only, ignores velocity!

    def __init__(self, arena_size=10, sample_rate=10, render_mode="rgb_array"):
        '''
        initializes a arena_size X arena_size square area
        '''
        super(InertialContinuousArena, self).__init__()
        self.render_mode = render_mode

        # Size of the 2D arena
        self.arena_size = arena_size

        # Sampling rate in Hz
        self.sample_rate=sample_rate
        
        # Initialize the agent near the lower-left
        self.agent_pos = [-arena_size*.9, -arena_size*.9]
        self.agent_vel = [0,0]

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have 5: thrust {up, down, left, right}, coast
        n_actions = 5
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be [x_pos, y_pos, x_vel, y_vel] of the agent
        low = np.array([-arena_size, -arena_size, -np.inf, -np.inf])
        high = np.array([arena_size, arena_size, np.inf, np.inf])
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(4,), dtype=np.float64
        )
        
        self.cum_reward = 0.

    def reset(self, seed=None, options=None, state=None):
        """
        Important: the observation must be a numpy array
        state: if None, initialize in lower-left. Otherwise, length-4 np array of initial state.
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        
        if state is None :
            # Initialize the agent randomly
            
            state = self.observation_space.sample()
            self.agent_pos = state[:2]
            self.agent_vel = state[2:]
            
            
            # self.agent_pos = np.array([-self.arena_size*.9, -self.arena_size*.9])
            # self.agent_vel = np.array([0,0])
            
        else :
            self.agent_pos = state[:2]
            self.agent_vel = state[2:]
            
        self.cum_reward = 0.
        
        return np.concatenate((self.agent_pos,self.agent_vel)), {}  # empty info dict

    def step(self, action):
        if action not in [0,1,2,3,4] :
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        
        self.agent_pos += self.agent_vel/self.sample_rate # propagate the position at the given sampling rate (before impulse)
        self.agent_vel += self.action_list[action] # add an impulse taken from the action_list (after propagating position)

        # Account for the boundaries of the grid: 0 agent's velocity in the component where it hit the wall?
        for dim in range(2) :
            if self.agent_pos[dim] > self.arena_size :
                self.agent_pos[dim] = self.arena_size
                self.agent_vel[dim] = 0 # if we hit the x (y) boundary, zero-out the x (y) velocity
            elif self.agent_pos[dim] < -self.arena_size :
                self.agent_pos[dim] = -self.arena_size
                self.agent_vel[dim] = 0 # if we hit the x (y) boundary, zero-out the x (y) velocity
                
        # self.agent_pos = np.clip(self.agent_pos, [-self.arena_size,-self.arena_size], [self.arena_size,self.arena_size])

        # Are we at the center of the grid?
        terminated = self.GOAL.contains(self.agent_pos)
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 0
        if action != self.COAST :
            reward -= 1 # thrust is costly
        if terminated :
            reward += 100 # getting to the goal is good
        else:
            reward -= 1/self.sample_rate # penalize delay (lose 1 reward per simulation second)

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        self.cum_reward += reward

        return (
            np.concatenate((self.agent_pos,self.agent_vel)),#.astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print("." * self.agent_pos, end="")
            print("x", end="")
            print("." * (self.grid_size - self.agent_pos))

        if self.render_mode == "rgb_array":
    
            # Unpack state
            pos = self.agent_pos
            vel = self.agent_vel
            vel_mag = np.linalg.norm(self.agent_vel)
        
            # Create a figure and manually set the canvas
            fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
            canvas = FigureCanvasAgg(fig)  # explicitly use Agg canvas
            fig.set_canvas(canvas)
        
            ax.set_xlim(-self.arena_size, self.arena_size)
            ax.set_ylim(-self.arena_size, self.arena_size)
            ax.set_aspect('equal')
            ax.axis("off")

            ax.plot(pos[0], pos[1], 'ro', markersize=5)
        
            # Draw the agent as an arrow (if vel is nonzero)
            if vel_mag > 1e-6:
                ax.arrow(
                    pos[0], pos[1],
                    vel[0], vel[1],
                    head_width=0.5,
                    head_length=0.7,
                    fc='blue',
                    ec='blue',
                    length_includes_head=True
                )
            
            # Draw arena border
            ax.plot([-self.arena_size, self.arena_size, self.arena_size, -self.arena_size, -self.arena_size],
                    [-self.arena_size, -self.arena_size, self.arena_size, self.arena_size, -self.arena_size], 'k-', linewidth=1)
    
            # Draw goal area border
            ax.plot([-1, 1, 1, -1, -1],
                    [-1, -1, 1, 1, -1], 'k-', linewidth=1)
        
            # Render to RGB array
            canvas.draw()
            buf = canvas.buffer_rgba()  # use RGBA buffer
            width, height = fig.get_size_inches() * fig.dpi
            image = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3]  # strip alpha
        
            plt.close(fig)
            return image


    def close(self):
        pass