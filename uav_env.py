###################################
## Environment Setup of for UAV  ##
################################### 

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import sys

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

class UAVenv(gym.Env):
    def __init__(self, args):
        super(UAVenv, self).__init__()
        self.USER_LOC = np.loadtxt('UserLocation.txt', delimiter=' ').astype(np.int64)
        self.USER_RB_REQ = np.loadtxt('UserRBReq.txt', delimiter=' ').astype(np.int64)
        
        # Environment specific params 
        self.args = args
        self.NUM_USER = self.args.num_user                      # Number of ground user
        self.NUM_UAV = self.args.num_uav                        # Number of UAV
        self.UAV_HEIGHT = self.args.uav_height
        self.grid_bound_x = self.args.grid_space                # square boundary 0 to grid_space
        self.grid_bound_y = self.args.grid_space
        self.rk_user_prob = self.args.rk_user_prob
        self.uav_max_connection = self.args.uav_max_connection
        self.dist_connection = self.args.dist_connection
        self.dist_detection = self.args.dist_detection
        self.done = 0
        self.current_state = None
        # Five different action for the movement of each UAV
        # 0 = Right, 1 = Left, 2 = straight, 3 = back, 4 = Hover

    def step(self, action):
        # move UAVs
        isDone = False
        for i in range(len(self.uav_id)):
            done = self.uav_class[i].move(action[i])
            if done:
                self.done += 1
            if self.done > 3:
                isDone = True
        for i in range(len(self.user_class)):
            rand = np.random.random()
            if rand < self.rk_user_prob:
                self.user_class[i].move()
        for i in range(len(self.user_class)):
            self.user_class[i].reconnect(self.uav_class, 
                                         self.dist_connection, 
                                         self.dist_detection, 
                                         self.uav_max_connection, 
                                         self.UAV_HEIGHT)
        # get the state of UAVs
        state_new = self.get_state()
        state_old = self.state
        self.state = state_new
        reward = self.calculate_reward(state_old, state_new)
        return state_old, state_new, isDone, reward

    def render(self, ax, mode='human', close=False):
        # Implement viz
        if mode == 'human':
            ax.cla()
            position = self.state[:, 0:2] * self.grid_space
            ax.scatter(self.u_loc[:, 0], self.u_loc[:, 1], c = '#ff0000', marker='o', label = "Users")
            ax.scatter(position[:, 0], position[:, 1], c = '#000000', marker='x', label = "UAV")
            for (i,j) in (position[:,:]):
                cc = plt.Circle((i,j), self.coverage_radius, alpha=0.1)
                ax.set_aspect(1)
                ax.add_artist(cc)
            ax.legend(loc="lower right")
            
            plt.pause(0.5)
            plt.xlim(-50, 1050)
            plt.ylim(-50, 1050)
            plt.draw()

    def reset(self):
        # Reset the environment to an initial state
        # get mobile users
        x_axis = np.random.random(self.NUM_USER) * self.grid_bound_x
        y_axis = np.random.random(self.NUM_USER) * self.grid_bound_y
        user_location = []
        for x, y in zip(x_axis, y_axis):
            user = MobileUser(x, y, 0, self.grid_bound_x, 0, self.grid_bound_y, 0, 1)
            user_location.append([user.x, user.y])
        self.user_class = user_location
        
        # get UAVs
        x_axis = np.random.random(self.NUM_UAV) * self.grid_bound_x
        y_axis = np.random.random(self.NUM_UAV) * self.grid_bound_y
        uav_location = []
        uav_id = []
        self.uav_available_min_id = 0
        for x, y in zip(x_axis, y_axis):
            uav = UAV(x, y, 0, self.grid_bound_x, 0, self.grid_bound_y, self.uav_available_min_id)
            uav_id.append(uav.id)
            self.uav_available_min_id += 1
            uav_location.append([uav.x, uav.y])
        self.uav_class = uav_location
        self.uav_id = uav_id

    def get_state(self):
        # Get the state of the environment
        state = []
        for i in range(len(self.uav_class)):
            uav = self.uav_class[i]
            state.append([uav.x, uav.y, uav.num_connected, sum(uav.detected_user_signal_strength)])
        return state
    
    def calculate_reward(self, state_old, state_new):
        rewards = []
        for i in range(len(state_old)):
            uav_old: UAV = state_old[i]
            uav_new: UAV = state_new[i]
            reward1 = uav_new.num_connected - uav_old.num_connected
            reward2 = sum(uav_new.detected_user_signal_strength) - sum(uav_old.detected_user_signal_strength)
            reward = reward1 + 100 * reward2
            rewards.append(reward)
        return rewards


class MobileUser:
    def __init__(self, x, y, x_min, x_max, y_min, y_max, d_min, d_max):
        self.x = x
        self.y = y
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.d_min = d_min
        self.d_max = d_max
        self.direction = random.uniform(0, 2 * math.pi)  # initial random direction
        self.connected_uav_id = None

    def move(self, direction_std=math.pi/8):
    # Update direction slightly (small random drift)
        self.direction += random.gauss(0, direction_std)

        # Choose random step size
        step_size = random.uniform(self.d_min, self.d_max)

        # Attempt new position
        x_new = self.x + step_size * math.cos(self.direction)
        y_new = self.y + step_size * math.sin(self.direction)

        # Reflect if out of bounds
        if x_new < self.x_min:
            x_new = 2 * self.x_min - x_new
            self.direction = math.pi - self.direction
        elif x_new > self.x_max:
            x_new = 2 * self.x_max - x_new
            self.direction = math.pi - self.direction

        if y_new < self.y_min:
            y_new = 2 * self.y_min - y_new
            self.direction = -self.direction
        elif y_new > self.y_max:
            y_new = 2 * self.y_max - y_new
            self.direction = -self.direction

        self.x = x_new
        self.y = y_new
        
        return self.x, self.y
    
    def get_position(self):
        return self.x, self.y
    
    def reconnect(self, list_uav, dist_c, dist_d, max_connection, uav_height):
        dist_list = []
        for uav in list_uav:
            distance = math.sqrt((self.x - uav.x) ** 2 + (self.y - uav.y) ** 2 + uav_height ** 2)
            dist_list.append(distance)
            if distance < dist_d:
                uav.detected_user_signal_strength.append(1 / distance ** 2)
        # Sort and argsort the dist_list
        sorted_indices = np.argsort(dist_list)
        dist_list = np.sort(dist_list)
        for idx, distance in zip(sorted_indices, dist_list):
            uav = list_uav[idx]
            if distance < dist_c and uav.num_connected < max_connection:
                uav.num_connected += 1
                self.connected_uav_id = uav.id
            


class UAV:
    def __init__(self, x, y, x_min, x_max, y_min, y_max, id, dist_c, dist_d):
        self.x = x
        self.y = y
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dist_connection = dist_c
        self.dist_detection = dist_d
        self.gaussian_noise = 0.5
        self.num_connected = 0
        self.id = id
        self.detected_user_signal_strength = []
    
    def move(self, action):
        # Action space
        # 0 = Right, 1 = Left, 2 = straight, 3 = back, 4 = Hover
        done = False
        if action == 0:
            self.x += (self.x_max - self.x_min) / 25 + random.gauss(0, self.gaussian_noise)
            if self.x > self.x_max:
                self.x = self.x_max
                done = True
        elif action == 1:
            self.x -= (self.x_max - self.x_min) / 25 + random.gauss(0, self.gaussian_noise)
            if self.x < self.x_min:
                self.x = self.x_min
                done = True
        elif action == 2:
            self.y += (self.y_max - self.y_min) / 25 + random.gauss(0, self.gaussian_noise)
            if self.y > self.y_max:
                self.y = self.y_max
                done = True
        elif action == 3:
            self.y -= (self.y_max - self.y_min) / 25 + random.gauss(0, self.gaussian_noise)
            if self.y < self.y_min:
                self.y = self.y_min
                done = True
        elif action == 4:
            pass
        else:
            print("Error Action Value")
        return done
    
    def clear_connections(self):
        self.num_connected = 0
        self.detected_user_signal_strength = []
