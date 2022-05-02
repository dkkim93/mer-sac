import torch
import numpy as np
from misc.rl_utils import initialize_debug, update_debug, log_debug


def train(agents, env, log, tb_writer, args):
    # Initialize game
    obs = env.reset()
    peer_latents = [torch.zeros((1, args.n_latent), device=agent.device) for agent in agents]
    debug = initialize_debug(agents, args, debug=None)

    for timestep in range(1, args.max_timestep):
        # Get joint action
        actions = []
        for agent, peer_latent in zip(agents, peer_latents):
            action = agent.act(obs, peer_latent, timestep)
            actions.append(action)
        update_debug(debug, args, agents=agents, obs=obs, peer_latents=peer_latents, actions=actions)

        # Take step in environment
        next_obs, rewards, done, _ = env.step(actions)
        next_obs = env.reset() if done else next_obs
        update_debug(debug, args, rewards=rewards)

        # Get next peer latent from encoder
        next_peer_latents = [
            agent.encode(peer_latent, obs, actions, reward, next_obs)
            for agent, peer_laent, reward in zip(agents, peer_latents, rewards)]

        # Add transition to memory
        for agent, peer_latent, reward, next_peer_latent in zip(agents, peer_latents, rewards, next_peer_latents):
            if "masac" in agent.name:
                agent.add_transition(
                    obs=obs,
                    peer_latent=peer_latents[1 - agent.i_agent],
                    actions=actions,
                    reward=rewards,
                    next_obs=next_obs,
                    next_peer_latent=next_peer_latents[1 - agent.i_agent])
            else:
                agent.add_transition(
                    obs=obs,
                    peer_latent=peer_latent,
                    actions=actions,
                    reward=reward,
                    next_obs=next_obs,
                    next_peer_latent=next_peer_latent)

        # Perform update for agent
        losses = []
        for agent in agents:
            losses.append(agent.get_loss(agents, timestep))

        for agent, loss in zip(agents, losses):
            agent.update(loss)

        # For next timestep
        obs = np.copy(next_obs)
        peer_latents = [next_peer_latent.clone().detach() for next_peer_latent in next_peer_latents]

        # For logging
        if done:
            log_debug(debug, agents, log, tb_writer, args, timestep)
            debug = initialize_debug(agents, args, debug=debug)
