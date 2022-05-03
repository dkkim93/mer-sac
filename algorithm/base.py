import abc
import torch
import numpy as np
import torch.nn.functional as F
from algorithm.replay_memory import ReplayMemory
from misc.rl_utils import get_env_dim, get_env_action_type
from torch.distributions import Normal


class Base(metaclass=abc.ABCMeta):
    """Base class for all algorithms
    Args:
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
        rank (int): Used for thread-specific meta-agent for multiprocessing. Default: -1
    """
    def __init__(self, env, log, tb_writer, args, name):
        super(Base, self).__init__()

        self.env = env
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name

    @abc.abstractmethod
    def get_loss(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    def _set_device(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_env_dim(self) -> None:
        self.env_n_obs, self.env_n_action = get_env_dim(self.env)
        self.log[self.args.log_name].info("[{}] Env input dim: {}".format(
            self.name, self.env_n_obs))
        self.log[self.args.log_name].info("[{}] Env output dim: {}".format(
            self.name, self.env_n_action))

    def _set_action_type(self) -> None:
        self.is_discrete_action, self.action_dtype = get_env_action_type(self.env)
        self.log[self.args.log_name].info("[{}] Discrete action space: {} with dtype {}".format(
            self.name, self.is_discrete_action, self.action_dtype))

        if not self.is_discrete_action:
            self.action_limit = self.env.action_space.high[0]
            self.log[self.args.log_name].info("[{}] Action limit: {}".format(
                self.name, self.action_limit))

    def _set_memory(self) -> None:
        self.memory = ReplayMemory(self.args, device=self.device)

    def _set_epsilon(self) -> None:
        self.epsilon = float(self.args.epsilon)

    def _select_action(self, mu, logvar):
        distribution = Normal(loc=mu, scale=torch.exp(logvar))
        action = distribution.rsample()
        action_logprob = distribution.log_prob(action).sum(axis=-1)
        action_logprob -= (2. * (np.log(2.) - action - F.softplus(-2. * action))).sum(axis=1)

        # Apply action limit
        action = self.action_limit * torch.tanh(action)

        return action, action_logprob

    def add_transition(self, obs, action, reward, next_obs, done) -> None:
        self.memory.push(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def encode(self, *args, **kwargs):
        return torch.zeros((1, self.args.n_latent), dtype=torch.float32, device=self.device)
