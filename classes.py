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
        low = np.array([-arena_size,    # x_pos
                        -arena_size,    # y_pos
                        -np.inf,        # x_vel
                        -np.inf])       # y_vel
        high = np.array([arena_size,    # x_pos
                         arena_size,    # y_pos
                         np.inf,        # x_vel
                         np.inf])       # y_vel
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
            self.agent_vel = state[2:4]
            
            
            # self.agent_pos = np.array([-self.arena_size*.9, -self.arena_size*.9])
            # self.agent_vel = np.array([0,0])
            
        else :
            self.agent_pos = state[:2]
            self.agent_vel = state[2:4]
            
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
    
    


class InertialContinuousArenaTrigger(InertialContinuousArena) :
    '''
    Extends the vanilla InertialContinuousArena by adding a "trigger time"
    This environment has 2 extra states: "T" and "TA"
    T: current simulation time (in seconds). Init to 0 and increments by 1/self.sample_rate every step
    TA: trigger time (in seconds): if agent enters target area when T<TA, penalty. If T>=TA, reward.
    '''
    def __init__(self, arena_size=10, sample_rate=10, render_mode="rgb_array", TA=5):
        '''
        initializes a arena_size X arena_size square area
        '''
        super(InertialContinuousArenaTrigger, self).__init__(arena_size,sample_rate,render_mode)
        
        self.TA = TA
        
        if TA>1e-6 : 
            self.TA_passed = 0.
        else :
            self.TA_passed = 1.
            
        self.T = 0.
        
        # The observation will be [x_pos, y_pos, x_vel, y_vel, T, TA_passed]
        low = np.concatenate((self.observation_space.low,
                              np.array([0.,0.])))   #T, TA_passed
        high = np.concatenate((self.observation_space.high,
                               np.array([np.inf,1.])))   #T, TA_passed
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(6,), dtype=np.float64
        )

    def reset(self, seed=None, options=None, state=None):
        """
        Important: the observation must be a numpy array
        state: if None, initialize in lower-left. Otherwise, length-4 np array of initial agent state.
        :return: (np.array)
        """
        obs,info = super().reset(seed=seed, options=options, state=state) # obs has no T, TA_passed
        # return obs
        
        self.T = 0. # set clock to 0
        
        if self.TA>1e-6 : 
            self.TA_passed = 0. 
        else :
            self.TA_passed = 1.
        
        # return obs, np.array((self.T,self.TA_passed))
        return np.concatenate((obs,np.array([self.T,self.TA_passed]))), info  # empty info dict



    def step(self, action):
        
        obs,rewardSuper,terminated,truncated,info = super().step(action)
        
        self.cum_reward -= rewardSuper
        
        # compute reward differently for trigger environment!
        # positive reward if terminated after TA, negative reward if terminated before TA
        
        self.T += 1/self.sample_rate # propagate time
        self.TA_passed = self.T >= self.TA-1e-6
        
        reward = 0.
        # reward -= np.linalg.norm(self.agent_pos)/10000 # slope the agent slightly toward the goal
        if action != self.COAST :
            reward -= 1 # thrust is always costly
        if self.TA_passed>0.5 : 
            if terminated :
                reward += 1000 # getting to the goal after TA is good
            else:
                reward -= 1/self.sample_rate # penalize delay after TA (lose 1 reward per simulation second)
        else :
            if terminated :
                reward -= 50 # getting to the goal before TA is bad
        
        self.cum_reward += reward
        
        obs = np.concatenate((obs,np.array([self.T,self.TA_passed]))) # create new obs

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
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
            
            
            # If TA_passed signal is active, display label
            ax.text(-9.5, 11.4, "T="+str(np.round(self.T,1)), fontsize=12, color='green', weight='bold')
            ax.text(-9.5, 10.2, "cum reward="+str(np.round(self.cum_reward,1)), fontsize=12, color='green', weight='bold')
            if self.TA_passed >= 0.5:
                ax.text(-9.5, 9, "TA Passed", fontsize=12, color='red', weight='bold')
    
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



if __name__=="__main__" :
    
    gym.register('InertialContinuousArenaTrigger',InertialContinuousArenaTrigger)
        
    # uglyVideo(10,env)
    
    
    # vec_env = make_vec_env(InertialContinuousArena, n_envs=1, env_kwargs=dict(arena_size=10))
    
    envTrig = InertialContinuousArenaTrigger()
    
    from stable_baselines3 import PPO
    from util import record_video
    
    model = PPO("MlpPolicy", envTrig, verbose=1, device='cpu')
    
    # record_video('InertialContinuousArenaTrigger', model,prefix='untrained')
    
    # Train the agent
    # model.learn(total_timesteps=10_000)
    # record_video('InertialContinuousArenaTrigger', model,prefix='10ksteps')
    model.learn(total_timesteps=100_000)
    record_video('InertialContinuousArenaTrigger', model,prefix='100ksteps')