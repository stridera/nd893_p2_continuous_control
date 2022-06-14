#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Train the given environment using Stable Baselines3 and PPO. """

import argparse
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from gym.spaces import Box
from unityagents import UnityEnvironment
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class UnityGymEnv(VecEnv):
    """Custom Environment that for UnityEnvironments"""
    metadata = {'render.modes': ['human']}

    def __init__(self, path: str, train_mode: bool = True):
        self.env = UnityEnvironment(file_name=path)
        self.train_mode = train_mode

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        # Env Details
        state_size = len(env_info.vector_observations[0])
        action_size = brain.vector_action_space_size
        self.num_agents = len(env_info.agents)

        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Number of agents: {self.num_agents}")

        self.action_space = Box(low=np.array(action_size * [-1.0]), high=np.array(action_size * [1.0]))
        self.observation_space = Box(
            low=np.array(state_size * [-float('inf')]),
            high=np.array(state_size * [float('inf')]))

        print(f'Action Space: {self.action_space.shape}')
        print(f'Obs Space: {self.observation_space.shape}')

        self.episode_steps = 0
        self.episode_rewards = np.zeros(self.num_agents)
        self.actions = None

        super().__init__(self.num_agents, self.observation_space, self.action_space)

    def step_async(self, actions) -> None:
        if self.num_agents:
            zero_padded_actions = np.zeros((self.num_agents, actions.shape[1]), dtype=np.float32)
            zero_padded_actions[:self.num_agents] = actions
            self.actions = zero_padded_actions
        else:
            self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        env_info = self.env.step(self.actions)[self.brain_name]
        states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished

        self.episode_rewards += rewards
        self.episode_steps += 1

        if any(dones):
            if not all(dones):
                logging.warning('All agent episodes were supposed to finish simultaneously, but this was not the case.'
                                f'{sum(dones)}/{len(dones)} agents done.')
            info = [
                dict(episode=dict(
                    r=self.episode_rewards[i],
                    l=self.episode_steps))
                for i in range(self.num_agents)
            ]
            self.reset()
        else:
            info = [dict() for _ in dones]

        return states,  np.array(rewards),  np.array(dones), info

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        self.episode_steps = 0
        self.episode_rewards = np.zeros(self.num_agents)

        return env_info.vector_observations

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def get_images(self, *args, **kwargs) -> Sequence[np.ndarray]:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        pass

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return super().env_is_wrapped(wrapper_class, indices)

    @property
    def n_envs(self) -> int:
        return self.num_agents


class InfoCallback(BaseCallback):

    def __init__(self, n_envs: int = 1, verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.episode = 0
        self.episode_scores = np.zeros(self.n_envs)
        self.last_100_scores = deque(maxlen=100)

    def _on_step(self) -> bool:
        self.episode_scores += self.locals['rewards']

        if any(self.locals['dones']):
            self.episode += 1
            score = np.mean(self.episode_scores)
            self.last_100_scores.append(score)
            self.logger.record('score/score', score)
            self.logger.record('score/last_100_scores', np.mean(self.last_100_scores))
            self.episode_scores = np.zeros(self.n_envs)

        return True


def main(env_path: str, train: bool = True, model_path: str = 'ppo_unity_env') -> None:
    print(f"Using environment: {env_path}")
    env = UnityGymEnv(env_path, train_mode=train)

    if train:
        print("Training model...")
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=model_path, name_prefix='ppo_')
        callback_list = [InfoCallback(env.n_envs), checkpoint_callback]
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./sb3_tensorboard_logs/")
        model.learn(total_timesteps=int(2e6),  callback=callback_list)
        model.save(model_path)
    else:
        print(f"Evaluating model {model_path}...")
        model = PPO.load(model_path)
        obs = env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if any(dones):
                obs = env.reset()

    env.close()


if __name__ == "__main__":
    default_path = 'unity_env/Reacher_Linux/Reacher.x86_64'
    parser = argparse.ArgumentParser(description='Stable Baseline3 Agent')
    parser.add_argument('--env', type=str, default=default_path,
                        help=f'Path to the environment.  Default: {default_path}')
    parser.add_argument('--train', action='store_true', default=False, help='Train or test')
    parser.add_argument('--model', type=str, default='ppo_unity_env',
                        help='Path to the model.  Default: ppo_unity_env')
    args = parser.parse_args()
    main(args.env, args.train, args.model)
