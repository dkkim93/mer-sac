import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.torch_utils import weight_init


class GaussianLSTM(nn.Module):
    """Actor network with LSTM that outputs action
    Args:
        n_input (int): Input dimension to network
        n_output (int): Output dimension of network
        name (str): Prefix for each layer
        log (dict): Dictionary that contains python logging
        args (argparse): Python argparse that contains arguments
        device (torch.device): Torch device to process tensors
    """
    def __init__(self, n_input, n_output, name, log, args, device) -> None:
        super(GaussianLSTM, self).__init__()

        self.name = name

        setattr(self, name + "_l1", nn.Linear(n_input, args.n_hidden, device=device))
        setattr(self, name + "_l2", nn.LSTM(args.n_hidden, args.n_hidden, batch_first=True, device=device))
        setattr(self, name + "_l3_mu", nn.Linear(args.n_hidden, n_output, device=device))
        setattr(self, name + "_l3_var", nn.Linear(args.n_hidden, n_output, device=device))
        self.apply(weight_init)
        log[args.log_name].info("[{}] {}".format(name, self))

    def forward(self, x):
        x, (hx, cx) = x
        assert len(x.shape) == 3, "Shape must be (n_batch, n_sequence, n_input)"

        # Compute LSTM output
        x = getattr(self, self.name + "_l1")(x)
        x = F.relu(x)
        if hx is None or cx is None:
            x, (hy, cy) = getattr(self, self.name + "_l2")(x)
        else:
            x, (hy, cy) = getattr(self, self.name + "_l2")(x, (hx, cx))

        # Compute mean and variance. For variance, we take the exponent so that it will
        # have anappropriate range [0, \infty]
        # Ref: https://towardsdatascience.com/beginner-guide-to-variational-autoencoders-vae-
        # with-pytorch-lightning-13dbc559ba4b
        mu = getattr(self, self.name + "_l3_mu")(x)
        var = torch.exp(0.5 * getattr(self, self.name + "_l3_var")(x))

        return mu, var, (hy, cy)
