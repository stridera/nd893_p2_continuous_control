""" Replay Buffer - Taken from the Udacity DDPG project. """
import numpy as np
import random
import torch
from collections import deque
from typing import NamedTuple, Tuple, List


class Experience(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int, device: str):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): device to use
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state: List[float], action: int, reward: float, next_state: float, done: float) -> None:
        """Add a new experience to memory.

        Params
        ======
            state (float): The initial state

        """
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> Tuple[torch.Tensor]:
        """Randomly sample a batch of experiences from memory.

        Returns
        =======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)
