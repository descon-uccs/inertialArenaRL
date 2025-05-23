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
        self.agent_pos = np.array([-arena_size*.9, -arena_size*.9])
        self.agent_vel = np.array([0,0])


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
        self.observation_space = spaces.Dict({'agent_state': 
                                              spaces.Box(
            low=low, high=high, shape=(4,), dtype=np.float64)}
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
            
            state = self.observation_space.sample()['agent_state']
            self.agent_pos = state[:2]
            self.agent_vel = state[2:4]
            
            
            # self.agent_pos = np.array([-self.arena_size*.9, -self.arena_size*.9])
            # self.agent_vel = np.array([0,0])
            
        else :
            state = np.array(state)
            self.agent_pos = state[:2]
            self.agent_vel = state[2:4]
            
        self.cum_reward = 0.
        
        return {'agent_state':np.concatenate((self.agent_pos,self.agent_vel))}, {}  # empty info dict


    def _reward(self,action,terminated,truncated) :
        
        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 0
        if action != self.COAST :
            reward -= 1 # thrust is costly
        if terminated :
            reward += 100 # getting to the goal is good
        else:
            reward -= 1/self.sample_rate # penalize delay (lose 1 reward per simulation second)
        
        return reward

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

        reward = self._reward(action,terminated,truncated)

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        self.cum_reward += reward

        return (
            {'agent_state': np.concatenate((self.agent_pos,self.agent_vel))},
            reward,
            terminated,
            truncated,
            info,
        )

    def _annotate_plot(self,ax) :
        '''
        ax: pyplot axes object to add stuff to!
        '''
        ax.text(-9.5, 10.2, "cum reward="+str(np.round(self.cum_reward,1)), fontsize=12, color='green', weight='bold')
        return ax

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
            
            ax = self._annotate_plot(ax)
        
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
            self.TA_passed = False
        else :
            self.TA_passed = True
            
        self.T = 0.
        
        self.observation_space['TA_passed'] = spaces.Discrete(2)
        # self.observation_space = spaces.Dict({
        #     "agent_state"   : state_observation_space,
        #     "TA_passed"     : spaces.Discrete(2)})
        

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
            self.TA_passed = False
        else :
            self.TA_passed = True
        
        # return obs, np.array((self.T,self.TA_passed))
        
        obs['TA_passed'] = self.TA_passed
        
        return obs,info

    def _reward(self,action,terminated,truncated) :
        
        # compute reward differently for trigger environment!
        # positive reward if terminated after TA, negative reward if terminated before TA
        reward = 0.
        # reward -= np.linalg.norm(self.agent_pos)/10000 # slope the agent slightly toward the goal
        if action != self.COAST :
            reward -= 1 # thrust is always costly
        if self.TA_passed : 
            if terminated :
                reward += 1000 # getting to the goal after TA is good
            else:
                reward -= 1/self.sample_rate # penalize delay after TA (lose 1 reward per simulation second)
        else :
            if terminated :
                reward -= 50 # getting to the goal before TA is bad
        return reward

    def step(self, action):
        
        self.T += 1/self.sample_rate # propagate time
        self.TA_passed = self.T >= self.TA-1e-6
        
        obs,reward,terminated,truncated,info = super().step(action) # super handles cumulative reward and calling the _reward method
        
        obs['TA_passed'] = self.TA_passed

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )
    
    def _annotate_plot(self,ax) :
        '''
        ax: pyplot axes object to add stuff to!
        This method is called from the parent's render() method and can be used
        to customize the display for the child's purposes.
        '''
        
        # If TA_passed signal is active, display label
        ax.text(-9.5, 11.4, "T="+str(np.round(self.T,1)), fontsize=12, color='green', weight='bold')
        ax.text(-9.5, 10.2, "cum reward="+str(np.round(self.cum_reward,1)), fontsize=12, color='green', weight='bold')
        if self.TA_passed :
            ax.text(-9.5, 9, "TA Passed", fontsize=12, color='red', weight='bold')
        return ax



class InertialContinuousArenaThrust(InertialContinuousArenaTrigger) :
    '''
    Extends the InertialContinuousArenaTrigger by modifying the 
    agent's thrust to be real-valued.
    Almost everything is the same except the action space is now a Box().
    '''
    def __init__(self, arena_size=10, sample_rate=10, render_mode="rgb_array", TA=5, maxThrust=1):
        '''
        initializes a arena_size X arena_size square area
        '''
        super(InertialContinuousArenaThrust, self).__init__(arena_size,sample_rate,render_mode)
        
        
        self.maxThrust = maxThrust
        
        # overwrite the self.action_space attribute:
        lowAction = np.array([-maxThrust]*2)
        highAction = np.array([maxThrust]*2)
        self.action_space = spaces.Box(low=lowAction, high=highAction, shape=(2,), dtype=np.float64)
        
        

    def reset(self, seed=None, options=None, state=None):
        """
        Important: the observation must be a numpy array
        state: if None, initialize in lower-left. Otherwise, length-4 np array of initial agent state.
        :return: (np.array)
        """
        obs,info = super().reset(seed=seed, options=options, state=state)
        
        return obs,info

    def _reward(self,action,terminated,truncated) :
        
        # compute reward differently for trigger environment!
        # positive reward if terminated after TA, negative reward if terminated before TA
        reward = 0.
        # reward -= np.linalg.norm(self.agent_pos)/10000 # slope the agent slightly toward the goal
        
        # the L1-norm of the impulse is a cost: (assume independent x and y thrusters I guess?)
        reward -= np.linalg.norm(action,ord=1)
        if self.TA_passed : 
            if terminated :
                reward += 1000 # getting to the goal after TA is good
            else:
                reward -= 1/self.sample_rate # penalize delay after TA (lose 1 reward per simulation second)
        else :
            if terminated :
                reward -= 50 # getting to the goal before TA is bad
        return reward


    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        
        
        self.T += 1/self.sample_rate # propagate time
        self.TA_passed = self.T >= self.TA-1e-6
        
        self.agent_pos += self.agent_vel/self.sample_rate # propagate the position at the given sampling rate (before impulse)
        self.agent_vel += action # add the action directly as an impulse

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

        reward = self._reward(action,terminated,truncated)

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        self.cum_reward += reward
        
        obs = {'agent_state': np.concatenate((self.agent_pos,self.agent_vel))}
        obs['TA_passed'] = self.TA_passed

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )
    
    def _annotate_plot(self,ax) :
        '''
        ax: pyplot axes object to add stuff to!
        This method is called from the parent's render() method and can be used
        to customize the display for the child's purposes.
        '''
        
        # If TA_passed signal is active, display label
        ax.text(-9.5, 11.4, "T="+str(np.round(self.T,1)), fontsize=12, color='green', weight='bold')
        ax.text(-9.5, 10.2, "cum reward="+str(np.round(self.cum_reward,1)), fontsize=12, color='green', weight='bold')
        if self.TA_passed :
            ax.text(-9.5, 9, "TA Passed", fontsize=12, color='red', weight='bold')
        return ax
    
    
    
class InertialThrustBudget(InertialContinuousArenaThrust) :
    '''
    Extends the InertialContinuousArenaThrust by adding a total L1 thrust budget.
    The simulation truncates 10 seconds after the fuel budget is exhausted.
    '''
    def __init__(self, arena_size=10, sample_rate=10, render_mode="rgb_array", TA=5, maxThrust=1, BR=10):
        '''
        initializes a arena_size X arena_size square area
        '''
        super(InertialContinuousArenaThrust, self).__init__(arena_size,sample_rate,render_mode)
        
        
        self.BR = BR
        lowBudget = np.array([0])
        highBudget = np.arrya([BR])
        self.observation_space['fuel_remaining'] = spaces.Box(lowBudget,
                                                              highBudget,
                                                              (1,),
                                                              dtype=np.float64)
        

    def reset(self, seed=None, options=None, state=None):
        """
        Important: the observation must be a numpy array
        state: if None, initialize in lower-left. Otherwise, length-4 np array of initial agent state.
        :return: (np.array)
        """
        obs,info = super().reset(seed=seed, options=options, state=state)
        
        return obs,info

    def _reward(self,action,terminated,truncated) :
        
        # compute reward differently for trigger environment!
        # positive reward if terminated after TA, negative reward if terminated before TA
        reward = 0.
        # reward -= np.linalg.norm(self.agent_pos)/10000 # slope the agent slightly toward the goal
        
        # the L1-norm of the impulse is a cost: (assume independent x and y thrusters I guess?)
        reward -= np.linalg.norm(action,ord=1)
        if self.TA_passed : 
            if terminated :
                reward += 1000 # getting to the goal after TA is good
            else:
                reward -= 1/self.sample_rate # penalize delay after TA (lose 1 reward per simulation second)
        else :
            if terminated :
                reward -= 50 # getting to the goal before TA is bad
        return reward


    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        
        
        self.T += 1/self.sample_rate # propagate time
        self.TA_passed = self.T >= self.TA-1e-6
        
        self.agent_pos += self.agent_vel/self.sample_rate # propagate the position at the given sampling rate (before impulse)
        self.agent_vel += action # add the action directly as an impulse

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

        reward = self._reward(action,terminated,truncated)

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        self.cum_reward += reward
        
        obs = {'agent_state': np.concatenate((self.agent_pos,self.agent_vel))}
        obs['TA_passed'] = self.TA_passed

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )
    
    def _annotate_plot(self,ax) :
        '''
        ax: pyplot axes object to add stuff to!
        This method is called from the parent's render() method and can be used
        to customize the display for the child's purposes.
        '''
        
        # If TA_passed signal is active, display label
        ax.text(-9.5, 11.4, "T="+str(np.round(self.T,1)), fontsize=12, color='green', weight='bold')
        ax.text(-9.5, 10.2, "cum reward="+str(np.round(self.cum_reward,1)), fontsize=12, color='green', weight='bold')
        if self.TA_passed :
            ax.text(-9.5, 9, "TA Passed", fontsize=12, color='red', weight='bold')
        return ax

if __name__=="__main__" :
    
    gym.register('InertialContinuousArenaThrust',InertialContinuousArenaThrust)
        
    # uglyVideo(10,env)
    
    
    # vec_env = make_vec_env(InertialContinuousArena, n_envs=1, env_kwargs=dict(arena_size=10))
    
    envThr = InertialContinuousArenaThrust()
    envThr.reset()
    
    from stable_baselines3 import PPO
    from util import record_video
    
    model = PPO("MultiInputPolicy", envThr, verbose=1, device='cpu')
    
    record_video('InertialContinuousArenaThrust', model,prefix='Thrust_untrained')
    
    # Train the agent
    model.learn(total_timesteps=10_000)
    record_video('InertialContinuousArenaThrust', model,prefix='Thrust_10ksteps')
    model.learn(total_timesteps=100_000)
    record_video('InertialContinuousArenaThrust', model,prefix='Thrust_100ksteps')