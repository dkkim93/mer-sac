import gym
import torch
import numpy as np
import torch.nn.functional as F

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


def initialize_debug(agents, args, debug):
    if args.env_name == "IMP-v0" and debug is not None:
        debug["rewards"] = np.zeros((args.n_agent,))
        debug["actions"] = np.zeros((args.n_agent, agents[0].env_n_action))
        debug["predictions"] = np.zeros((args.n_agent,))
    else:
        debug = {
            "rewards": np.zeros((len(agents),)),
            "accumulate_rewards": np.zeros((len(agents),)),
            "actions": np.zeros((len(agents), agents[0].env_n_action)),
            "predictions": np.zeros((len(agents),))}

    return debug


def update_debug(debug, args, **kwargs):
    for key, value in kwargs.items():
        if "rewards" in key:
            debug["rewards"] += np.array(value).flatten()
            debug["accumulate_rewards"] += np.array(value).flatten()

        if "actions" in key:
            for i_agent, action in enumerate(value):
                debug["actions"][i_agent, action[0]] += 1

        if "agents" in key:
            for i_agent, agent in enumerate(value):
                if "further" in agent.name or "lili" in agent.name:
                    # Perform decoding
                    obs = torch.tensor(kwargs["obs"], dtype=torch.float32, device=agent.device)
                    peer_latent = kwargs["peer_latents"][i_agent]
                    decoder_input = torch.cat([obs, peer_latent], dim=-1)
                    decoder_out = agent.decoder(decoder_input)
                    peer_prob = F.softmax(decoder_out, dim=-1)

                    # Get prediction
                    peer_action = kwargs["actions"][1 - agent.i_agent][0]
                    peer_action_pred = torch.argmax(peer_prob, dim=-1)[-1].detach().cpu().numpy()
                    debug["predictions"][i_agent] += int(np.equal(peer_action, peer_action_pred))


def log_debug(debug, agents, log, tb_writer, args, timestep):
    """Log performance of training at each task
    Args:
        scores (list): Contains scores for each agent
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        timestep (int): timestep of training
    """
    for i_agent in range(len(agents)):
        reward = debug["rewards"][i_agent] / args.ep_horizon
        log[args.log_name].info("Episodic reward: {:.3f} for agent {} at timestep {}".format(
            reward, i_agent, timestep))
        tb_writer.add_scalars("debug/reward", {"agent" + str(i_agent): reward}, timestep)

    if "IMP-v0" in args.env_name:
        for i_agent in range(args.n_agent):
            accumulate_reward = debug["accumulate_rewards"][i_agent]
            log[args.log_name].info("Accumulate reward: {:.3f} for agent {} at timestep {}".format(
                accumulate_reward, i_agent, timestep))
            tb_writer.add_scalars("debug/accumulate_reward", {"agent" + str(i_agent): accumulate_reward}, timestep)

    for i_agent in range(len(agents)):
        for i_action in range(agents[i_agent].env_n_action):
            action = debug["actions"][i_agent, i_action] / args.ep_horizon
            log[args.log_name].info("{} Action: {:.3f} and for agent {} at iteration {}".format(
                i_action, action, i_agent, timestep))
            tb_writer.add_scalars("debug/action/agent" + str(i_agent), {"action" + str(i_action): action}, timestep)

    for i_agent in range(len(agents)):
        for i_action in range(agents[i_agent].env_n_action):
            prediction = debug["predictions"][i_agent] / args.ep_horizon
            tb_writer.add_scalars("debug/prediction", {"agent" + str(i_agent): prediction}, timestep)


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
