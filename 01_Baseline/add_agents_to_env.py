import numpy as np
from agents_in_env import RED
from generate_agents_in_env import allocate_red_pos


def add_reds(reds, config, battlefield, blues, config_add_agents):
    """ Add red agents during the episode """

    num_platoons_add = config_add_agents.num_red_platoons_add
    num_companies_add = config_add_agents.num_red_companies_add
    efficiencies_add = config_add_agents.efficiencies_red_add  # reds efficiency range

    agents = []
    for _ in range(num_platoons_add):
        agent = RED(agent_type='platoon', config=config)
        agent.force = config.threshold * config.mul + \
                      (config.agent_forces[0] - config.threshold * config.mul) * np.random.rand()
        agents.append(agent)

    for _ in range(num_companies_add):
        agent = RED(agent_type='company', config=config)
        agent.force = config.agent_forces[0] + \
                      (config.agent_forces[1] - config.threshold * config.mul) * np.random.rand()
        agents.append(agent)

    for idx, agent in enumerate(agents):
        agent.id = agent.color + '_' + str(idx + config.num_red_agents)

        agent.efficiency = \
            efficiencies_add[0] + (efficiencies_add[1] - efficiencies_add[0]) * np.random.rand()
        agent.ef = agent.force * agent.efficiency

        agent.effective_force = agent.force - agent.threshold
        agent.effective_ef = agent.ef - agent.threshold * agent.efficiency

        agent.initial_force = agent.force
        agent.initial_ef = agent.ef

        agent.initial_effective_force = agent.effective_force
        agent.initial_effective_ef = agent.effective_ef

    # allocate adding agent's initial position
    for agent in agents:
        agent.pos = allocate_red_pos(battlefield, blues, config.offset)

    return agents
