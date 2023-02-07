import numpy as np


class ConfigAddAgents:
    def __init__(self):
        # Define possible adding red agent parameters
        self.red_platoons_add = (4, 4)  # num range of red platoons, default=(3,7)
        self.red_companies_add = (1, 1)  # num range of red companies, default=(1,3)

        self.efficiencies_red_add = (0.3, 0.5)  # range

        # self.num_red_agents_add = None  # set in 'define_red_team'
        self.num_red_platoons_add = None
        self.num_red_companies_add = None

    def define_red_add_team(self):
        self.num_red_platoons_add = np.random.randint(
            low=self.red_platoons_add[0],
            high=self.red_platoons_add[1] + 1
        )

        self.num_red_companies_add = np.random.randint(
            low=self.red_companies_add[0],
            high=self.red_companies_add[1] + 1
        )

    def reset(self, config):
        self.define_red_add_team()

        if (self.num_red_platoons_add + self.num_red_companies_add) > \
                (config.max_num_red_agents - config.num_red_platoons - config.num_red_companies):

            raise ValueError()
