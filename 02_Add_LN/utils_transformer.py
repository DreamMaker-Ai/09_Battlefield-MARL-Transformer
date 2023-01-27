import numpy as np
import copy


def make_mask(alive_agents_ids, max_num_agents):
    mask = np.zeros(max_num_agents)  # (n,)

    for i in range(len(alive_agents_ids)):
        mask[i] = 1

    mask = mask.astype(bool)  # (n,)
    mask = np.expand_dims(mask, axis=0)  # add batch_dim, (1,n)

    return mask


def make_padded_obs(max_num_agents, obs_shape, raw_obs):
    padded_obs = copy.deepcopy(raw_obs)  # list of raw obs
    padding = np.zeros(obs_shape)  # 0-padding of obs

    while len(padded_obs) < max_num_agents:
        padded_obs.append(padding)

    padded_obs = np.stack(padded_obs)  # stack to sequence (agent) dim  (n,g,g,ch*n_frames)

    padded_obs = np.expand_dims(padded_obs, axis=0)  # add batch_dim (1,n,g,g,ch*n_frames)

    return padded_obs


def make_next_states_for_q(alive_agents_ids, next_alive_agents_ids, raw_next_states, obs_shape):
    """
    Make the next states list to compute Q(s',a') bases on alive agents ids

    :param alive_agents_ids: list, len=a=num_alive_agents
    :param next_alive_agents_ids: list, len =a'=num_next_alive_agents
    :param raw_next_states: lsit of obs, len=a'
    :param obs_shape: (g,g,ch*n_frames)
    :return: list of padded next_states_for_q, corresponding to the alive_agents_list, len=n,
    """

    next_states_for_q = []
    padding = np.zeros(obs_shape)

    for idx in alive_agents_ids:
        if idx in next_alive_agents_ids:
            next_states_for_q.append(raw_next_states[next_alive_agents_ids.index(idx)])
        else:
            next_states_for_q.append(padding)

    return next_states_for_q


def main():
    alive_agents_ids = [0, 2, 3, 4, 5, 7]
    next_alive_agents_ids = [2, 5]
    obs_shape = (2, 2)
    raw_next_states = [np.ones(obs_shape) * 2, np.ones(obs_shape) * 5]

    next_states_for_q = \
        make_next_states_for_q(
            alive_agents_ids=alive_agents_ids,
            next_alive_agents_ids=next_alive_agents_ids,
            raw_next_states=raw_next_states,
            obs_shape=obs_shape,
        )

    next_padded_states_for_q = \
        make_padded_obs(max_num_agents=10, obs_shape=obs_shape, raw_obs=next_states_for_q)

    print(next_padded_states_for_q)

    max_num_agents = 10
    mask = make_mask(alive_agents_ids, max_num_agents)
    print(mask)


if __name__ == '__main__':
    main()
