import torch
import numpy as np
from collections import namedtuple, deque


class ReplayMemory(object):
    """Simple replay memory that contains trajectories for each task
    in a Markov chain
    Args:
        args (argparse): Python argparse that contains arguments
    Refs:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    def __init__(self, args, device):
        self.memory = deque([], maxlen=args.memory_capacity)
        self.transition = namedtuple(
            'transition', ("obs", "peer_latent", "actions", "reward", "next_obs", "next_peer_latent"))
        self.args = args
        self.device = device

    def push(self, obs, peer_latent, actions, reward, next_obs, next_peer_latent):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        peer_latent = peer_latent.clone().detach()

        actions = torch.tensor(np.array(actions, dtype=np.float32).reshape(1, -1), dtype=torch.float32, device=self.device)

        if not isinstance(reward, np.ndarray):
            reward = np.array(reward, dtype=np.float32)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(1, -1)

        if not isinstance(next_obs, np.ndarray):
            next_obs = np.array(next_obs, dtype=np.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        next_peer_latent = next_peer_latent.clone().detach()

        self.memory.append(self.transition(*(obs, peer_latent, actions, reward, next_obs, next_peer_latent)))

    def sample(self, mode):
        if mode == "random":
            indices = np.random.randint(0, len(self), size=self.args.batch_size)
        elif mode == "recent":
            if len(self) < self.args.batch_size:
                indices = np.arange(len(self))
            else:
                indices = np.arange(len(self) - self.args.batch_size, len(self))
        elif mode == "now":
            indices = [len(self) - 1]
        else:
            raise ValueError("Invalid sampling mode")

        batch = self.transition(*zip(*[self.memory[index] for index in indices]))
        obs = torch.cat(batch.obs, dim=0)
        peer_latent = torch.cat(batch.peer_latent, dim=0)
        actions = torch.cat(batch.actions, dim=0)
        reward = torch.cat(batch.reward, dim=0)
        next_obs = torch.cat(batch.next_obs, dim=0)
        next_peer_latent = torch.cat(batch.next_peer_latent, dim=0)

        return obs, peer_latent, actions, reward, next_obs, next_peer_latent

    def __len__(self):
        return len(self.memory)
