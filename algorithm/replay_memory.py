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
        self.transition = namedtuple("transition", ("obs", "action", "reward", "next_obs", "done"))
        self.args = args
        self.device = device

    def push(self, obs, action, reward, next_obs, done):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        action = torch.tensor(np.array(action, dtype=np.float32).reshape(1, -1), dtype=torch.float32, device=self.device)

        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(1, -1)

        if not isinstance(next_obs, np.ndarray):
            next_obs = np.array(next_obs, dtype=np.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        done = torch.tensor(done, dtype=torch.float32, device=self.device).reshape(1, -1)

        self.memory.append(self.transition(*(obs, action, reward, next_obs, done)))

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
        action = torch.cat(batch.action, dim=0)
        reward = torch.cat(batch.reward, dim=0)
        next_obs = torch.cat(batch.next_obs, dim=0)
        done = torch.cat(batch.done, dim=0)

        return obs, action, reward, next_obs, done

    def __len__(self):
        return len(self.memory)
