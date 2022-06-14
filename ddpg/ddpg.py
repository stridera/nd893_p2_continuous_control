""" DDPG """
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from .model import Actor, Critic
from .ou_noise import OUNoise
from .replay_buffer import Experience, ReplayBuffer


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic


class Agent():
    """
    Deep Deterministic Policy Gradient (DDPG): Deep Q-Learning for a Continuous Control Problem.
    https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(self, state_size: int, action_size: int, num_agents: int = 1, seed: int = 0,
                 writer: SummaryWriter = None) -> None:
        """ Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        random.seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks - Q Network with an Actor and a Critic
        self.actor_local = Actor(state_size, action_size, seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device)

        # Initialize time step (for updating tensorboard)
        self.t_step = 0
        self.writer = writer

    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Save experience in replay memory, and use random sample from buffer to learn.

        Params
        ======
            state (np.ndarray): current state
            action (np.ndarray): action taken
            reward (float): reward received
            next_state (np.ndarray): next state
            done (bool): if the episode is done
        """
        self.t_step += 1
        self.memory.add(state, action, reward, next_state, done)

    def invoke_training(self):
        """Train the model on the target experiences."""
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state: np.ndarray, add_noise: bool = True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (np.ndarray): current state
            add_noise (bool): add noise to actions
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experience: Experience, gamma: float):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experience

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------- record losses  -------------------------- #
        if self.writer is not None:
            self.writer.add_scalar('loss/critic', critic_loss, self.t_step)
            self.writer.add_scalar('loss/actor', actor_loss, self.t_step)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Params
        ======
            path (str): path to save the model
        """
        os.makedirs(path, exist_ok=False)
        torch.save(self.actor_local.state_dict(), path + 'actor.pth')
        torch.save(self.critic_local.state_dict(), path + 'critic.pth')
        print("Model Saved.")

    def load_model(self, path: str) -> None:
        """Load the model from a file.

        Params
        ======
            path (str): path to load the model
        """
        self.actor_local.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic_local.load_state_dict(torch.load(path + 'critic.pth'))
        print("Model Loaded.")
