import torch
import copy
import itertools
import numpy as np
import torch.nn.functional as F
from algorithm.base import Base
from torch.distributions import Normal
from network.categorical_mlp import CategoricalMLP
from network.gaussian_mlp import GaussianMLP
from misc.torch_utils import get_parameters, to_numpy
from misc.rl_utils import to_transition, reparameterization


class FURTHER(Base):
    """Class for FURTHER agent
    Args:
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
    """
    def __init__(self, env, log, tb_writer, args, name, i_agent) -> None:
        super(FURTHER, self).__init__(env, log, tb_writer, args, name, i_agent)

        self._set_device()
        self._set_env_dim()
        self._set_action_type()
        self._set_actor()
        self._set_critic()
        self._set_gain()
        self._set_inference()
        self._set_optimizer()
        self._set_memory()

    def _set_actor(self) -> None:
        self.actor = GaussianMLP(
            n_input=self.env_n_obs + self.args.n_latent,
            n_output=self.env_n_action, name=self.name + "_actor",
            log=self.log, args=self.args, device=self.device)

    def _set_critic(self) -> None:
        self.critic1 = CategoricalMLP(
            n_input=self.env_n_obs + self.args.n_latent + self.args.n_agent * self.env_n_action,
            n_output=1,
            name=self.name + "_critic1",
            log=self.log, args=self.args, device=self.device)
        self.critic_target1 = copy.deepcopy(self.critic1)

        self.critic2 = CategoricalMLP(
            n_input=self.env_n_obs + self.args.n_latent + self.args.n_agent * self.env_n_action,
            n_output=1,
            name=self.name + "_critic2",
            log=self.log, args=self.args, device=self.device)
        self.critic_target2 = copy.deepcopy(self.critic2)

    def _set_gain(self) -> None:
        gain = torch.tensor(0., dtype=torch.float32, device=self.device)
        self.gain = torch.nn.Parameter(gain, requires_grad=True)

    def _set_inference(self) -> None:
        self.encoder = GaussianMLP(
            n_input=self.env_n_obs * 2 + self.args.n_agent * self.env_n_action + 1 + self.args.n_latent,
            n_output=self.args.n_latent,
            name=self.name + "_encoder",
            log=self.log, args=self.args, device=self.device)

        self.decoder = GaussianMLP(
            n_input=self.env_n_obs + self.args.n_latent,
            n_output=self.env_n_action,
            name=self.name + "_decoder",
            log=self.log, args=self.args, device=self.device)

    def _set_optimizer(self) -> None:
        # Set actor optimizer
        self.actor_optimizer = torch.optim.Adam(get_parameters(self.actor), lr=self.args.actor_lr)

        # Set critic optimizer
        critic_params = itertools.chain(get_parameters(self.critic1), get_parameters(self.critic2))
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.args.critic_lr)

        # Set gain optimizer
        self.gain_optimizer = torch.optim.Adam(get_parameters(self.gain), lr=self.args.gain_lr)

        # Set inference optimizer
        inference_params = itertools.chain(get_parameters(self.encoder), get_parameters(self.decoder))
        self.inference_optimizer = torch.optim.Adam(inference_params, lr=self.args.inference_lr)

    def encode(self, peer_latent, obs, actions, reward, next_obs):
        transition = to_transition(obs, actions, reward, next_obs, self, self.args)
        encoder_input = torch.cat([peer_latent, transition], dim=-1)
        encoder_mu, encoder_logvar = self.encoder(encoder_input)
        next_peer_latent = reparameterization(encoder_mu, encoder_logvar)
        return next_peer_latent

    def act(self, obs, peer_latent, timestep):
        # Compute output of policy
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor_input = torch.cat([obs, peer_latent], dim=-1)
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
        obs, peer_latent, actions, _, _, _ = self.memory.sample(mode="random")

        # Get log_prob
        actor_input = torch.cat([obs, peer_latent], dim=-1).detach()
        actor_mu, actor_logvar = self.actor(actor_input)
        action, action_logprob = self._select_action(actor_mu, actor_logvar)

        # Get Q-value
        actions[:, self.i_agent * self.env_n_action:(self.i_agent + 1) * self.env_n_action] = action
        critic_input = torch.cat([obs, peer_latent, actions], dim=-1)
        q_value1 = self.critic1(critic_input)
        q_value2 = self.critic2(critic_input)
        q_value = torch.minimum(q_value1, q_value2).squeeze(1)

        # Get actor loss
        assert action_logprob.shape == q_value.shape, "{} vs {}".format(action_logprob.shape, q_value.shape)
        self.loss["actor"] = (self.args.entropy_weight * action_logprob - q_value).mean()

        # For logging
        self.tb_writer.add_scalars(
            "loss/entropy", {"agent" + str(self.i_agent): -action_logprob.mean()}, timestep)

    def _get_critic_loss(self, timestep):
        if timestep < self.args.batch_size:
            _, _, _, reward, _, _ = self.memory.sample(mode="recent")
            if torch.max(reward) > self.gain:
                self.gain.data = torch.max(reward)
            self.loss["critic"] = 0.
            return

        # Process transition
        obs, peer_latent, actions, reward, next_obs, _ = self.memory.sample(mode="random")
        next_peer_latent = self.encode(peer_latent, obs, actions, reward, next_obs)

        # Get Q-value
        critic_input = torch.cat([obs, peer_latent, actions], dim=-1).detach()
        q_value1 = self.critic1(critic_input)
        q_value2 = self.critic2(critic_input)

        # Get own agent's next action probability
        next_actor_input = torch.cat([next_obs, next_peer_latent], dim=-1)
        next_actor_mu, next_actor_logvar = self.actor(next_actor_input)
        next_action, next_action_logprob = self._select_action(next_actor_mu, next_actor_logvar)

        # Get next_peer_action
        decoder_input = torch.cat([next_obs, next_peer_latent], dim=-1)
        next_decoder_mu, next_decoder_logvar = self.decoder(decoder_input)
        next_peer_action, _ = self._select_action(next_decoder_mu, next_decoder_logvar)

        # Get next Q-value
        if self.i_agent == 0:
            next_actions = torch.cat([next_action, next_peer_action], dim=-1)
        else:
            next_actions = torch.cat([next_peer_action, next_action], dim=-1)

        next_critic_input = torch.cat([next_obs, next_peer_latent, next_actions], dim=-1).detach()
        next_q_value1 = self.critic_target1(next_critic_input)
        next_q_value2 = self.critic_target2(next_critic_input)
        next_q_value = torch.minimum(next_q_value1, next_q_value2)

        # Get critic loss
        next_value = next_q_value - self.args.entropy_weight * next_action_logprob.reshape(-1, 1)
        target = reward - self.gain + next_value.detach()
        assert q_value1.shape == target.shape, "{} vs {}".format(q_value1.shape, target.shape)
        self.loss["critic"] = torch.square(q_value1 - target).mean() + torch.square(q_value2 - target).mean()

        # For logging
        self.tb_writer.add_scalars("debug/gain", {"agent" + str(self.i_agent): self.gain}, timestep)

    def _get_inference_loss(self, timestep):
        if timestep < self.args.batch_size:
            self.loss["inference"], self.loss["kl"] = 0., 0.
            return

        # Set initial prior flag
        is_initial_prior = True if timestep == self.args.batch_size else False

        # Perform encoding
        obs, peer_latent, actions, reward, next_obs, _ = self.memory.sample(mode="recent")

        if self.args.inference_train_mode == "batch":
            encoder_input = torch.cat([peer_latent, obs, actions, reward, next_obs], dim=-1)[:-1, :]
            encoder_mu, encoder_logvar = self.encoder(encoder_input)
            next_peer_latent = reparameterization(encoder_mu, encoder_logvar)

        elif self.args.inference_train_mode == "sequential":
            peer_latent_ = peer_latent[0, :]
            encoder_mu, encoder_logvar, next_peer_latent = [], [], []
            for timestep in range(obs.shape[0]):
                if timestep == int(obs.shape[0] - 1):
                    break

                encoder_input_ = torch.cat([
                    peer_latent_, obs[timestep, :], actions[timestep, :],
                    reward[timestep, :], next_obs[timestep, :]], dim=-1).detach()
                encoder_mu_, encoder_logvar_ = self.encoder(encoder_input_)
                next_peer_latent_ = reparameterization(encoder_mu_, encoder_logvar_)

                encoder_mu.append(encoder_mu_)
                encoder_logvar.append(encoder_logvar_)
                next_peer_latent.append(next_peer_latent_)

                # For next timestep
                peer_latent_ = next_peer_latent_.clone().detach()
            next_peer_latent = torch.stack(next_peer_latent)
            encoder_mu, encoder_logvar = torch.stack(encoder_mu), torch.stack(encoder_logvar)

        else:
            raise ValueError("Invalid inference train mode")

        # Get reconstruction loss
        decoder_input = torch.cat([obs[1:, :], next_peer_latent], dim=-1)
        decoder_mu, decoder_logvar = self.decoder(decoder_input)
        distribution = Normal(loc=decoder_mu, scale=torch.exp(decoder_logvar))
        next_peer_action = actions[1:, (1 - self.i_agent) * self.env_n_action:(2 - self.i_agent) * self.env_n_action]
        assert distribution.sample().shape == next_peer_action.shape
        next_peer_action_logprob = distribution.log_prob(next_peer_action).sum(axis=-1)
        next_peer_action_logprob -= (2. * (np.log(2.) - next_peer_action - F.softplus(-2. * next_peer_action))).sum(axis=1)
        self.loss["inference"] = -torch.mean(next_peer_action_logprob)

        # Get KLD error
        if is_initial_prior:
            encoder_mu = torch.cat([torch.zeros((1, self.args.n_latent), device=self.device), encoder_mu], dim=0)
            encoder_logvar = torch.cat([torch.zeros((1, self.args.n_latent), device=self.device), encoder_logvar], dim=0)

        kl_first_term = torch.sum(encoder_logvar[:-1, :], dim=-1) - torch.sum(encoder_logvar[1:, :], dim=-1)
        kl_second_term = self.args.n_latent
        kl_third_term = torch.sum(1. / torch.exp(encoder_logvar[:-1, :]) * torch.exp(encoder_logvar[1:, :]), dim=-1)
        kl_fourth_term = \
            (encoder_mu[:-1, :] - encoder_mu[1:, :]) / torch.exp(encoder_logvar[:-1, :]) * \
            (encoder_mu[:-1, :] - encoder_mu[1:, :])
        kl_fourth_term = kl_fourth_term.sum(dim=-1)
        kl = 0.5 * (kl_first_term - kl_second_term + kl_third_term + kl_fourth_term)
        self.loss["kl"] = self.args.kl_weight * torch.mean(kl)

    def get_loss(self, agents, timestep):
        # Initialize loss
        self.loss = {}

        # Get actor and inference loss
        self._get_actor_loss(timestep)
        self._get_critic_loss(timestep)
        self._get_inference_loss(timestep)

        # For logging
        for key, value in self.loss.items():
            self.tb_writer.add_scalars("loss/" + key, {"agent" + str(self.i_agent): value}, timestep)

        return self.loss

    def update(self, loss):
        if isinstance(loss["actor"], torch.Tensor):
            self.actor_optimizer.zero_grad()
            loss["actor"].backward()
            torch.nn.utils.clip_grad_norm_(get_parameters(self.actor), self.args.max_grad_clip)
            self.actor_optimizer.step()

        if isinstance(loss["critic"], torch.Tensor):
            self.critic_optimizer.zero_grad()
            self.gain_optimizer.zero_grad()
            loss["critic"].backward()
            critic_params = itertools.chain(get_parameters(self.critic1), get_parameters(self.critic2))
            torch.nn.utils.clip_grad_norm_(critic_params, self.args.max_grad_clip)
            torch.nn.utils.clip_grad_norm_(get_parameters(self.gain), self.args.max_grad_clip)
            self.critic_optimizer.step()
            self.gain_optimizer.step()

        if isinstance(loss["inference"], torch.Tensor) or isinstance(loss["kl"], torch.Tensor):
            self.inference_optimizer.zero_grad()
            (loss["inference"] + loss["kl"]).backward()
            inference_params = itertools.chain(get_parameters(self.encoder), get_parameters(self.decoder))
            torch.nn.utils.clip_grad_norm_(inference_params, self.args.max_grad_clip)
            self.inference_optimizer.step()

        with torch.no_grad():
            for p, p_target in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                p_target.data.mul_(self.args.polyak)
                p_target.data.add_((1. - self.args.polyak) * p.data)

            for p, p_target in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                p_target.data.mul_(self.args.polyak)
                p_target.data.add_((1. - self.args.polyak) * p.data)
