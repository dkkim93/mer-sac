import gym
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_env_dim(env):
    if isinstance(env.observation_space, gym.spaces.Box):
        env_n_obs = env.observation_space.shape[0]
    else:
        env_n_obs = env.observation_space.n

    if isinstance(env.action_space, gym.spaces.Box):
        env_n_action = env.action_space.shape[0]
    else:
        env_n_action = env.action_space.n

    return env_n_obs, env_n_action


def get_env_action_type(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete_action = True
        action_dtype = np.int64
    else:
        is_discrete_action = False
        action_dtype = np.float32

    return is_discrete_action, action_dtype


def to_transition(obs, actions, reward, next_obs, agent, args):
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)

    if not isinstance(actions, torch.Tensor):
        actions = np.array(actions, dtype=np.int64).reshape(1, -1)
        actions = torch.tensor(actions, dtype=torch.int64, device=agent.device)
    actions_onehot = [
        to_onehot(actions[..., i_agent], dim=agent.env_n_action)
        for i_agent in range(args.n_agent)]
    actions_onehot = torch.cat(actions_onehot, dim=-1).float()

    if not isinstance(reward, torch.Tensor):
        reward = torch.tensor(reward, dtype=torch.float32, device=agent.device).unsqueeze(1)

    if not isinstance(next_obs, torch.Tensor):
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)

    return torch.cat([obs, actions_onehot, reward, next_obs], dim=-1)


def initialize_debug():
    debug = {"reward": 0.}
    return debug


def update_debug(debug, args, **kwargs):
    for key, value in kwargs.items():
        if "reward" in key:
            debug["reward"] += value


def log_debug(debug, agent, log, tb_writer, args, timestep):
    """Log performance of training at each task
    Args:
        scores (list): Contains scores for each agent
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        timestep (int): timestep of training
    """
    reward = debug["reward"]
    log[args.log_name].info("Episodic reward: {:.3f} at timestep {}".format(reward, timestep))
    tb_writer.add_scalar("debug/reward", reward, timestep)


def to_onehot(value, dim):
    """Convert batch of tensor numbers to onehot
    Args:
        value (numpy.ndarray or torch.Tensor): Batch of numbers to convert to onehot
        dim (int): Dimension of onehot
    Returns:
        onehot (numpy.ndarray or torch.Tensor): Converted onehot
    References:
        https://gist.github.com/NegatioN/acbd8bb6be866ce1831b2d073fd7c450
    """
    if isinstance(value, np.ndarray):
        assert len(value.shape) == 1, "Shape must be (batch,)"
        onehot = np.eye(dim, dtype=np.float32)[value]
        assert onehot.shape == (value.shape[0], dim), "Shape must be: (batch, dim)"

    elif isinstance(value, torch.Tensor):
        if value.dtype != torch.int64:
            raise ValueError("Used?")
            value = value.int64()
        scatter_dim = len(value.size())
        y_tensor = value.view(*value.size(), -1)
        zeros = torch.zeros(*value.size(), dim, dtype=value.dtype, device=value.device)
        onehot = zeros.scatter(scatter_dim, y_tensor, 1)

    else:
        raise ValueError("Not supported data type")

    return onehot


def reparameterization(mean, logvar):
    var = torch.exp(0.5 * logvar)
    distribution = torch.distributions.Normal(mean, var)
    z = distribution.rsample()
    return z
