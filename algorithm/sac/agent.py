import torch
import copy
import itertools
from algorithm.base import Base
from network.categorical_mlp import CategoricalMLP
from network.gaussian_mlp import GaussianMLP
from misc.torch_utils import get_parameters, to_numpy
from misc.rl_utils import to_transition, reparameterization


class SAC(Base):
    """Class for SAC agent
    Args:
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
    """
    def __init__(self, env, log, tb_writer, args, name) -> None:
        super(SAC, self).__init__(env, log, tb_writer, args, name)

        self._set_device()
        self._set_env_dim()
        self._set_action_type()
        self._set_actor()
        self._set_critic()
        self._set_optimizer()
        self._set_memory()

    def _set_actor(self) -> None:
        self.actor = GaussianMLP(
            n_input=self.env_n_obs,
            n_output=self.env_n_action, name=self.name + "_actor",
            log=self.log, args=self.args, device=self.device)

    def _set_critic(self) -> None:
        self.critic1 = CategoricalMLP(
            n_input=self.env_n_obs + self.env_n_action,
            n_output=1,
            name=self.name + "_critic1",
            log=self.log, args=self.args, device=self.device)
        self.critic_target1 = copy.deepcopy(self.critic1)

        self.critic2 = CategoricalMLP(
            n_input=self.env_n_obs + self.env_n_action,
            n_output=1,
            name=self.name + "_critic2",
            log=self.log, args=self.args, device=self.device)
        self.critic_target2 = copy.deepcopy(self.critic2)

    def _set_optimizer(self) -> None:
        # Set actor optimizer
        self.actor_optimizer = torch.optim.Adam(get_parameters(self.actor), lr=self.args.actor_lr)

        # Set critic optimizer
        critic_params = itertools.chain(get_parameters(self.critic1), get_parameters(self.critic2))
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.args.critic_lr)

    def encode(self, peer_latent, obs, actions, reward, next_obs):
        transition = to_transition(obs, actions, reward, next_obs, self, self.args)
        encoder_input = torch.cat([peer_latent, transition], dim=-1)
        encoder_mu, encoder_logvar = self.encoder(encoder_input)
        next_peer_latent = reparameterization(encoder_mu, encoder_logvar)
        return next_peer_latent

    def act(self, obs):
        # Compute output of policy
        actor_input = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor_mu, actor_logvar = self.actor(actor_input)

        # Select action
        action, _ = self._select_action(actor_mu, actor_logvar)
        action = to_numpy(action.flatten(), dtype=self.action_dtype)

        return action

    def _get_actor_loss(self, timestep):
        if timestep < self.args.batch_size:
            self.loss["actor"] = 0.
            return

        # Process transition
        obs, action, _, _, _ = self.memory.sample(mode="random")

        # Get log_prob
        actor_mu, actor_logvar = self.actor(obs)
        action, action_logprob = self._select_action(actor_mu, actor_logvar)

        # Get Q-value
        critic_input = torch.cat([obs, action], dim=-1)
        q_value1 = self.critic1(critic_input)
        q_value2 = self.critic2(critic_input)
        q_value = torch.minimum(q_value1, q_value2).squeeze(1)

        # Get actor loss
        assert action_logprob.shape == q_value.shape, "{} vs {}".format(action_logprob.shape, q_value.shape)
        self.loss["actor"] = (self.args.entropy_weight * action_logprob - q_value).mean()

        # For logging
        self.tb_writer.add_scalar("loss/entropy", -action_logprob.mean(), timestep)

    def _get_critic_loss(self, timestep):
        if timestep < self.args.batch_size:
            self.loss["critic"] = 0.
            return

        # Process transition
        obs, action, reward, next_obs, done = self.memory.sample(mode="random")

        # Get Q-value
        critic_input = torch.cat([obs, action], dim=-1).detach()
        q_value1 = self.critic1(critic_input)
        q_value2 = self.critic2(critic_input)

        # Get own agent's next action probability
        next_actor_mu, next_actor_logvar = self.actor(next_obs)
        next_action, next_action_logprob = self._select_action(next_actor_mu, next_actor_logvar)

        # Get next Q-value
        next_critic_input = torch.cat([next_obs, next_action], dim=-1).detach()
        next_q_value1 = self.critic_target1(next_critic_input)
        next_q_value2 = self.critic_target2(next_critic_input)
        next_q_value = torch.minimum(next_q_value1, next_q_value2)

        # Get critic loss
        next_value = next_q_value - self.args.entropy_weight * next_action_logprob.reshape(-1, 1)
        target = reward + (1. - done) * self.args.discount * next_value.detach()
        assert q_value1.shape == target.shape, "{} vs {}".format(q_value1.shape, target.shape)
        self.loss["critic"] = torch.square(q_value1 - target).mean() + torch.square(q_value2 - target).mean()

    def get_loss(self, agents, timestep):
        # Initialize loss
        self.loss = {}

        # Get actor and inference loss
        self._get_actor_loss(timestep)
        self._get_critic_loss(timestep)

        # For logging
        for key, value in self.loss.items():
            self.tb_writer.add_scalar("loss/" + key, value, timestep)

        return self.loss

    def update(self, loss):
        if isinstance(loss["actor"], torch.Tensor):
            self.actor_optimizer.zero_grad()
            loss["actor"].backward()
            torch.nn.utils.clip_grad_norm_(get_parameters(self.actor), self.args.max_grad_clip)
            self.actor_optimizer.step()

        if isinstance(loss["critic"], torch.Tensor):
            self.critic_optimizer.zero_grad()
            loss["critic"].backward()
            critic_params = itertools.chain(get_parameters(self.critic1), get_parameters(self.critic2))
            torch.nn.utils.clip_grad_norm_(critic_params, self.args.max_grad_clip)
            self.critic_optimizer.step()

        with torch.no_grad():
            for p, p_target in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                p_target.data.mul_(self.args.polyak)
                p_target.data.add_((1. - self.args.polyak) * p.data)

            for p, p_target in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                p_target.data.mul_(self.args.polyak)
                p_target.data.add_((1. - self.args.polyak) * p.data)
