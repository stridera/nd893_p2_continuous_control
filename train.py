#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Train the given environment using PPO. """

import argparse
from collections import deque
from typing import Optional
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

from ddpg import Agent


def main(env: str, episodes: int = 100, seed: Optional[int] = None, max_steps: int = 1000) -> None:
    print(f"Using environment: {env}")
    env = UnityEnvironment(file_name=env)
    seed = seed if seed is not None else np.random.randint(0, 10000)

    writer = SummaryWriter()

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    print(f"Seed: {seed}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Number of agents: {num_agents}")

    agent = Agent(state_size, action_size, seed=seed, writer=writer)

    ep_avg_scores = deque(maxlen=100)
    progress = tqdm(range(1, episodes + 1), desc="Training", ncols=120)
    solved = False
    for i_episode in progress:
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()

        for step in range(max_steps):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)

            if step % 20 == 0:
                for _ in range(10):
                    agent.invoke_training()

            states = next_states
            scores += rewards
            if np.any(dones):
                break

        avg_score = np.mean(scores)
        ep_avg_scores.append(avg_score)
        writer.add_scalar("score/score", avg_score, i_episode)
        writer.add_scalar("score/last_100_scores", np.mean(ep_avg_scores), i_episode)

        print(f'Episode {i_episode}\tAverage Score: {np.mean(avg_score):.2f}\tScore: {avg_score:.2f}', ' ' * 120)

        if i_episode % 10 == 0:
            print(f"\rEpisode {i_episode} Last 100 avg score: {np.mean(ep_avg_scores)}                                ")
            agent.save_model(f"models/checkpoint_{i_episode}/")

        progress.set_postfix(avg_score=avg_score, last_100_avg=np.mean(ep_avg_scores))

        if not solved and np.mean(ep_avg_scores) > 30.0:
            print(
                f"Solved in {i_episode} episodes with a last 100 episode avg score of {np.mean(ep_avg_scores)}!")
            agent.save_model("models/solved/")
            solved = True

    # Save final model
    print(f"Final score of last 100 episodes: {np.mean(ep_avg_scores)}")
    agent.save_model("models/final/")

    env.close()
    writer.close()


if __name__ == "__main__":
    default_path = 'unity_env/Reacher_Linux/Reacher.x86_64'
    parser = argparse.ArgumentParser(description='DQN Agent')
    parser.add_argument('--env', type=str, default=default_path,
                        help=f'Path to the environment.  Default: {default_path}')
    parser.add_argument('--episodes', type=int, default=600, help='Number of episodes to run.  Default: 600')
    parser.add_argument('--seed', type=int, required=False, help='Random seed. Default: None (random)')

    args = parser.parse_args()
    main(args.env, args.episodes, args.seed)
