import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


def weight_init(module):
    """Initialize layer weight based on Xavier normal
    Only supported layer types are nn.Linear and nn.LSTMCell

    Args:
        module (class): Layer to initialize weight, including bias
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        module.bias.data.zero_()

    if isinstance(module, nn.LSTMCell):
        nn.init.xavier_normal_(module.weight_ih)
        nn.init.xavier_normal_(module.weight_hh)
        module.bias_ih.data.zero_()
        module.bias_hh.data.zero_()


def to_numpy(tensor, dtype):
    return tensor.detach().cpu().numpy().astype(dtype)


def get_parameters(network):
    """Return parameters that consist of network

    Args:
        network (class): Network that consists of torch parameters or variables

    Returns:
       parameters (generator): Set of parameters that consist of network
    """
    if isinstance(network, OrderedDict):
        return network.values()
    elif isinstance(network, nn.Parameter):
        return [network]
    elif isinstance(network, torch.Tensor):
        return [network]
    else:
        return network.parameters()


def get_named_parameters(network):
    """Return named_parameters that consist of network

    Args:
        network (class): Network that consists of torch parameters or variables

    Returns:
       named parameters (generator): Set of parameters with names that consist of network
    """
    if isinstance(network, OrderedDict):
        return network.items()
    else:
        return network.named_parameters()


def zero_grad(network):
    """Zero gradient in network

    Args:
        network (class): Network that consists of torch parameters or variables
    """
    if isinstance(network, torch.nn.Parameter):
        if network.grad is not None:
            network.grad.zero_()
    else:
        network.zero_grad()


def ensure_shared_grads(model, shared_model, gpu=False):
    """Ensure shared gradients between a thread-specific model and shared model

    Args:
        model (class): A thread-specific model
        shared_model (class): Shared model across threads

    References:
        https://github.com/dgriff777/rl_a3c_pytorch/blob/master/utils.py
    """
    if isinstance(model, torch.nn.Parameter) and isinstance(shared_model, torch.nn.Parameter):
        shared_model._grad = model.grad.cpu() if gpu else model.grad
    else:
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None and not gpu:
                return
            elif not gpu:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()


def change_name(network, old, new):
    """Change network's old name with new name

    Args:
        network (class): Network that consists of torch parameters or variables
        old (str): Original name in network
        new (str): Name to change to
    """
    new_network = {}
    for name, param in get_named_parameters(network):
        name = name.replace(old, new)
        new_network[name] = param
    return new_network


def save_checkpoint(agent, args, iteration):
    path = \
        "./pretrain_model/" + args.env_name + "/" + \
        "actor::" + str(agent.i_agent) + "_" + \
        "reward_scale::" + str(args.reward_scale) + "_" + \
        "entropy_weight::" + str(args.entropy_weight) + "_" + \
        "iteration::" + str(iteration) + ".pth"
    checkpoint = {'actor_state_dict': agent.actor.state_dict()}
    torch.save(checkpoint, path)


def load_checkpoint(i_agent, args, train_iteration):
    iteration = np.random.randint(low=0, high=train_iteration + 1)
    path = \
        "./pretrain_model/" + args.env_name + "/" + \
        "actor::" + str(i_agent) + "_" + \
        "reward_scale::" + str(args.reward_scale) + "_" + \
        "entropy_weight::" + str(args.entropy_weight) + "_" + \
        "iteration::" + str(iteration) + ".pth"
    return torch.load(path)
