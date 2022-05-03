import numpy as np
from misc.rl_utils import initialize_debug, update_debug, log_debug


def train(agent, env, log, tb_writer, args):
    # Initialize game
    obs = env.reset()
    debug = initialize_debug()

    for timestep in range(1, args.max_timestep):
        # Get action
        action = agent.act(obs)

        # Take step in environment
        next_obs, reward, done, _ = env.step(action)
        update_debug(debug, args, reward=reward)

        # Add transition to memory
        agent.add_transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

        # Perform update for agent
        loss = agent.get_loss(agent, timestep)
        agent.update(loss)

        # For next timestep
        obs = np.copy(next_obs)

        # For logging
        if done:
            obs = env.reset()
            log_debug(debug, agent, log, tb_writer, args, timestep)
            debug = initialize_debug()
