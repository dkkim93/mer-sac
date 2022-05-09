def get_agent(env, log, tb_writer, args, agent_type):
    if agent_type == "sac":
        from algorithm.sac import SAC as Algorithm
    elif agent_type == "baseline":
        from algorithm.baseline import Baseline as Algorithm
    elif agent_type == "baseline2":
        from algorithm.baseline2 import Baseline2 as Algorithm
    else:
        raise ValueError("Not supported algorithm: {}".format(agent_type))

    agent = Algorithm(env=env, log=log, tb_writer=tb_writer, args=args, name=agent_type)

    return agent
