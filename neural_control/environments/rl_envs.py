import gym
import math
import numpy as np
import torch
import time
from gym import spaces
import cv2
from gym.utils import seeding

from neural_control.environments.drone_env import QuadRotorEnvBase
from neural_control.trajectory.q_funcs import project_to_line
from neural_control.dataset import WingDataset, QuadDataset
from neural_control.trajectory.generate_trajectory import (
    load_prepare_trajectory
)
from neural_control.trajectory.random_traj import PolyObject
metadata = {'render.modes': ['human']}

buffer_len = 3
img_width, img_height = (200, 300)
crop_width = 60
center_at_x = True


class QuadEnvRL(QuadRotorEnvBase, gym.Env):

    def __init__(self, dynamics, dt, speed_factor=.2, horizon=10, **kwargs):
        self.dt = dt
        self.speed_factor = speed_factor
        self.horizon = horizon

        QuadRotorEnvBase.__init__(self, dynamics, dt)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4, ))

        # state and reference
        self.state_inp_dim = 15
        self.obs_dim = self.state_inp_dim + horizon * 9
        high = np.array([10 for _ in range(self.obs_dim)])
        self.observation_space = spaces.Box(
            -high, high, shape=(self.obs_dim, )
        )

        self.thresh_stable = 1.5
        self.thresh_div = .3

        kwargs["dt"] = dt
        kwargs['speed_factor'] = speed_factor
        kwargs["self_play"] = 0
        self.dataset = QuadDataset(1, **kwargs)

    def prepare_obs(self):
        obs_state, _, obs_ref, _ = self.dataset.prepare_data(
            self.state.copy(),
            self.current_ref[self.current_ind + 1:self.current_ind +
                             self.horizon + 1].copy()
        )
        return obs_state, obs_ref

    def state_to_obs(self):
        # get from dataset
        obs_state, obs_ref = self.prepare_obs()
        # flatten obs ref
        obs_ref = obs_ref.reshape((-1, self.obs_dim - self.state_inp_dim))
        # concatenate relative position and observation
        obs = torch.cat((obs_ref, obs_state), dim=1)[0].numpy()
        return obs

    def reset(self, test=0):
        # load random trajectory from train
        self.current_ref = load_prepare_trajectory(
            "data/traj_data_1", self.dt, self.speed_factor, test=test
        )
        self.renderer.add_object(PolyObject(self.current_ref))

        self.state = np.zeros(12)
        self.state[:3] = self.current_ref[0, :3]
        self._state.from_np(self.state)

        self.current_ind = 0
        self.obs = self.state_to_obs()
        return self.obs

    def get_divergence(self):
        return np.linalg.norm(
            self.current_ref[self.current_ind, :3] - self.state[:3]
        )

    def get_reward_mpc(self, action):
        """
        MPC type cost function turned into reward
        """
        pos_factor = 10
        u_thrust_factor = 5
        u_rates_factor = 0.1
        av_factor = 0.1
        vel_factor = 1

        pos_div = np.linalg.norm(
            self.current_ref[self.current_ind, :3] - self.state[:3]
        )
        pos_rew = self.thresh_div - pos_div
        vel_div = np.linalg.norm(
            self.current_ref[self.current_ind, 6:9] - self.state[6:9]
        )
        vel_rew = self.thresh_div - vel_div  # How high is velocity diff?
        u_rew = .5 - np.absolute(.5 - action)
        # have to use abs because otherwise not comparable to thresh div
        av_rew = np.sum(self.thresh_stable - (np.absolute(self.state[9:12])))

        # print()
        reward = .1 * (
            pos_factor * pos_rew + vel_factor * vel_rew + av_factor * av_rew +
            u_rates_factor * np.sum(u_rew[1:]) + u_thrust_factor * u_rew[0]
        )

        return reward

    def get_reward_mario(self, action):
        """
        ori_coeff: -0.01        # reward coefficient for orientation
        ang_vel_coeff: 0   # reward coefficient for angular velocity
        # epsilon coefficient
        pos_epsilon: 2        # reward epsilon for position 
        ori_epsilon: 0.2        # reward epsilon for orientation
        lin_vel_epsilon: 2   # reward epsilon for linear velocity
        ang_vel_epsilon: 0.2   # reward epsilon for angular velocity
        """
        pos_coeff = -0.02
        pos_epsilon = 2
        lin_vel_coeff = -0.002
        lin_vel_epsilon = 2
        survive_reward = 0.1  #  0.001 mario
        act_coeff = -0.001
        ori_coeff = -0.01  # ori_coeff: -0.01
        omega_coefficient = -0.001
        omega_epsilon = 2
        ori_epsilon = .2

        # position
        pos_loss = np.sum(
            self.current_ref[self.current_ind, :3] - self.state[:3]
        )**2
        pos_reward = pos_coeff * (pos_loss - pos_epsilon)
        # orientation:
        ori_loss = np.sum(
            self.current_ref[self.current_ind, 3:6] - self.state[3:6]
        )**2
        ori_reward = ori_coeff * (ori_loss - ori_epsilon)
        # velocity
        vel_loss = np.sum(
            self.current_ref[self.current_ind, 6:9] - self.state[6:9]
        )**2
        vel_reward = lin_vel_coeff * (vel_loss - lin_vel_epsilon)
        # body rates
        # omega_loss = np.sum(
        #     self.current_ref[self.current_ind, 9:] - self.state[9:]
        # )**2
        # omega_reward = omega_coefficient * (omega_loss - omega_epsilon)
        # action
        act_reward = act_coeff * np.sum((.5 - action)**2)

        # print(
        #     "pos", pos_reward, "vel", vel_reward, "survive", survive_reward,
        #     "act", act_reward, "ori", ori_reward, "omega", omega_reward
        # )
        return (
            pos_reward + vel_reward + survive_reward + act_reward +
            ori_reward  # + omega_reward
        )

    def set_state(self, state):
        self.state = state
        self._state.from_np(state)

    def step(self, action):
        # rescale action
        action = (action + 1) / 2
        self.state, is_stable = QuadRotorEnvBase.step(
            self, action, thresh=self.thresh_stable
        )
        self.obs = self.state_to_obs()
        self.current_ind += 1

        pos_div = self.get_divergence()

        done = (
            (not is_stable) or pos_div > self.thresh_div
            or self.current_ind > len(self.current_ref) - self.horizon - 2
        )

        reward = 0
        if not done:
            # reward = self.thresh_div - pos_div
            # reward = self.get_reward(action)
            reward = self.get_reward_mario(action)
        info = {}
        # print()
        # np.set_printoptions(precision=3, suppress=1)
        # # print(self.current_ref[:3, :3])
        # print(
        #     self.current_ind, self.state[:3],
        #     self.current_ref[self.current_ind, :3]
        # )
        # print(self.state)
        # print(self.obs.shape)
        # print(div, reward)
        return self.obs, reward, done, info

    def render(self, mode="human"):
        self._state.position[2] += 1
        QuadRotorEnvBase.render(self, mode=mode)
        self._state.position[2] -= 1
        time.sleep(self.dt)



class QuadEnvMario(QuadEnvRL):

    def __init__(self, dynamics, dt, speed_factor=.5, horizon=1):
        super().__init__(
            dynamics, dt, speed_factor=speed_factor, horizon=horizon
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(4, ))

        # state and reference
        self.state_inp_dim = 15
        self.obs_dim = 27
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim, )
        )
