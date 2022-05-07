import numpy as np
from misc.rl_utils import initialize_debug, update_debug, log_debug


def train(agent, env_reg, env_opp, log, tb_writer, args):
    ############################################################
    # Initialize game
    obs = env_reg.reset()
    debug = initialize_debug()

    for timestep in range(1, args.max_timestep):
        # Get action
        action = agent.act(obs)

        # Take step in environment
        next_obs, reward, done, _ = env_reg.step(action)
        update_debug(debug, args, reward=reward)

        # Add transition to memory
        agent.add_transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, stage="reg")

        # Perform update for agent
        loss = agent.get_loss(agent, timestep, stage="reg")
        agent.update(loss)

        # For next timestep
        obs = np.copy(next_obs)

        # For logging
        if done:
            obs = env_reg.reset()
            log_debug(debug, agent, log, tb_writer, args, timestep)
            debug = initialize_debug()

    ############################################################
    # Initialize game
    obs = env_opp.reset()
    debug = initialize_debug()
    for timestep in range(args.max_timestep + 1, 2 * args.max_timestep):
        # Get action
        action = agent.act(obs)

        # Take step in environment
        next_obs, reward, done, _ = env_opp.step(action)
        update_debug(debug, args, reward=reward)

        # Add transition to memory
        agent.add_transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, stage="opp")

        # Perform update for agent
        loss = agent.get_loss(agent, timestep, stage="opp")
        agent.update(loss)

        # For next timestep
        obs = np.copy(next_obs)

        # For logging
        if done:
            obs = env_opp.reset()
            log_debug(debug, agent, log, tb_writer, args, timestep)
            debug = initialize_debug()

    ############################################################
    # Initialize game
    obs = env_reg.reset()
    debug = initialize_debug()
    for timestep in range(2 * args.max_timestep + 1, 3 * args.max_timestep):
        # Get action
        action = agent.act(obs)

        # Take step in environment
        next_obs, reward, done, _ = env_reg.step(action)
        update_debug(debug, args, reward=reward)

        # Add transition to memory
        agent.add_transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, stage="reg")

        # Perform update for agent
        loss = agent.get_loss(agent, timestep, stage="reg")
        agent.update(loss)

        # For next timestep
        obs = np.copy(next_obs)

        # For logging
        if done:
            obs = env_reg.reset()
            log_debug(debug, agent, log, tb_writer, args, timestep)
            debug = initialize_debug()
