def get_agent(env, log, tb_writer, args, agent_type, i_agent):
    if agent_type == "further":
        from algorithm.further import FURTHER as Algorithm
    elif agent_type == "lili":
        from algorithm.lili import LILI as Algorithm
    elif agent_type == "masac":
        from algorithm.masac import MASAC as Algorithm
    else:
        raise ValueError("Not supported algorithm: {}".format(agent_type))

    agent = Algorithm(env=env, log=log, tb_writer=tb_writer, args=args, name=agent_type, i_agent=i_agent)

    return agent
