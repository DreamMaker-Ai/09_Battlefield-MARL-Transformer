import random

import ray
import gym
import tensorflow as tf
import numpy as np
from collections import deque

from battlefield_strategy import BattleFieldStrategy

from models import MarlTransformerModel
from utils_gnn import get_alive_agents_ids
from utils_transformer import make_mask, make_padded_obs


def who_wins(result_red, result_blue):
    num_alive_reds = result_red['num_alive_agent']
    num_alive_blues = result_blue['num_alive_agent']

    if (num_alive_reds <= 0) & (num_alive_blues <= 0):
        winner = 'draw'

    elif num_alive_reds <= 0:
        winner = 'blue_win'

    elif num_alive_blues <= 0:
        winner = 'red_win'

    else:
        winner = 'no_contest'

    return winner


def summarize_agent_result(agents):
    num_platoon = 0
    num_company = 0
    num_alive_platoon = 0
    num_alive_company = 0
    remaining_effective_force = 0
    initial_effective_force = 0

    for agent in agents:
        if agent.type == 'platoon':
            num_platoon += 1
        else:
            num_company += 1

        if agent.alive:
            if agent.type == 'platoon':
                num_alive_platoon += 1
            else:
                num_alive_company += 1

            remaining_effective_force += agent.effective_force

        initial_effective_force += agent.initial_effective_force

    num_initial_agents = num_platoon + num_company
    num_alive = num_alive_platoon + num_alive_company

    result = {}

    result['num_initial_agent'] = num_initial_agents
    result['num_initial_platoon'] = num_platoon
    result['num_initial_company'] = num_company

    result['num_alive_agent'] = num_alive
    result['num_alive_platoon'] = num_alive_platoon
    result['num_alive_company'] = num_alive_company

    result['initial_effective_force'] = initial_effective_force
    result['remaining_effective_force'] = remaining_effective_force

    return result


def summarize_episode_results(results, result_red, result_blue, winner):
    # For reds
    results['alive_reds_ratio'].append(
        result_red['num_alive_agent'] / result_red['num_initial_agent']
    )

    results['alive_red_platoon'].append(result_red['num_alive_platoon'])
    results['alive_red_company'].append(result_red['num_alive_company'])

    results['initial_red_platoon'].append(result_red['num_initial_platoon'])
    results['initial_red_company'].append(result_red['num_initial_company'])

    results['remaining_red_effective_force_ratio'].append(
        result_red['remaining_effective_force'] / result_red['initial_effective_force']
    )

    # For blues
    results['alive_blues_ratio'].append(
        result_blue['num_alive_agent'] / result_blue['num_initial_agent']
    )

    results['alive_blue_platoon'].append(result_blue['num_alive_platoon'])
    results['alive_blue_company'].append(result_blue['num_alive_company'])

    results['initial_blue_platoon'].append(result_blue['num_initial_platoon'])
    results['initial_blue_company'].append(result_blue['num_initial_company'])

    results['remaining_blue_effective_force_ratio'].append(
        result_blue['remaining_effective_force'] / result_blue['initial_effective_force']
    )

    results['winner'].append(winner)

    return results


def summarize_results(results):
    result = {}

    result['episode_rewards'] = np.mean(results['episode_rewards'])
    result['episode_lens'] = np.mean(results['episode_lens'])

    result['num_alive_reds_ratio'] = np.mean(results['alive_reds_ratio'])
    result['num_alive_red_platoon'] = np.mean(results['alive_red_platoon'])
    result['num_alive_red_company'] = np.mean(results['alive_red_company'])
    result['remaining_red_effective_force_ratio'] = \
        np.mean(results['remaining_red_effective_force_ratio'])

    result['num_initial_red_platoon'] = np.mean(results['initial_red_platoon'])
    result['num_initial_red_company'] = np.mean(results['initial_red_company'])

    result['num_alive_blues_ratio'] = np.mean(results['alive_blues_ratio'])
    result['num_alive_blue_platoon'] = np.mean(results['alive_blue_platoon'])
    result['num_alive_blue_company'] = np.mean(results['alive_blue_company'])
    result['remaining_blue_effective_force_ratio'] = \
        np.mean(results['remaining_blue_effective_force_ratio'])

    result['num_initial_blue_platoon'] = np.mean(results['initial_blue_platoon'])
    result['num_initial_blue_company'] = np.mean(results['initial_blue_company'])

    result['num_red_win'] = results['winner'].count('red_win')
    result['num_blue_win'] = results['winner'].count('blue_win')
    result['draw'] = results['winner'].count('draw')
    result['no_contest'] = results['winner'].count('no_contest')

    return result


# @ray.remote
@ray.remote(num_cpus=2, num_gpus=0)
class Tester:
    def __init__(self):
        # Make a copy of environment
        self.env = BattleFieldStrategy()
        self.action_space_dim = self.env.action_space.n
        self.n_frames = self.env.config.n_frames
        self.epsilon = 0.0
        self.obs_shape = (self.env.config.grid_size,
                          self.env.config.grid_size,
                          self.env.config.observation_channels * self.n_frames)

        # Make a q_network
        self.policy = MarlTransformerModel(config=self.env.config)

        # Initialize environment
        ### The followings are reset in 'reset_states'
        self.frames = None  # For each agent in env
        self.states = None  # Stack frames n_frames times
        self.prev_actions = None

        self.alive_agents_ids = None  # For all agents, including dummy ones
        self.padded_states = None
        self.padded_prev_actions = None
        self.mask = None

        # self.episode_reward = None
        self.step = None

        ### Initialize above Nones
        observations = self.env.reset()
        self.reset_states(observations)

        # make_animation
        if self.env.config.make_animation:
            self.env.make_animation.add_frame(self.env)
            red = self.env.reds[-1]
            if red.alive:
                self.env.make_animation.add_observations_3(observations[red.id])

    def reset_states(self, observations):
        # TODO prev_actions
        """
        alive_agents_ids: list of alive agent id

        # For agents in Env
             each agent stacks observations n-frames in channel-dims
             -> observations[red.id]: (grid_size,grid_size,channels)

             -> generate deque of length=n_frames
             self.frames[red.id]: deque[(grid_size,grid_size,channels),...]

             -> transform to states
             states[red.id]: (grid_size,grid_size,channels*n_frames)

             self.prev_actions[red.id]: int (TODO)
        """

        self.frames = {}
        self.states = {}
        self.prev_actions = {}

        for red in self.env.reds:
            # all reds are alive when reset

            self.frames[red.id] = deque([observations[red.id]] * self.n_frames,
                                        maxlen=self.n_frames)
            # [(grid_size,grid_size,channels),...,(grid_size,grid_size,channels)]

            self.states[red.id] = np.concatenate(self.frames[red.id], axis=2).astype(np.float32)
            # (grid_size,grid_size,channels*n_frames)

            # self.prev_actions[red.id] = 0

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.alive_agents_ids = get_alive_agents_ids(env=self.env)

        # Get raw observation list of alive agents
        raw_states = []

        for a in self.alive_agents_ids:
            agent_id = 'red_' + str(a)
            raw_states.append(self.states[agent_id])  # append (g,g,ch*n_frames)

        # Get padded observations ndarray for all agents, including dead and dummy agents
        self.padded_states = \
            make_padded_obs(max_num_agents=self.env.config.max_num_red_agents,
                            obs_shape=self.obs_shape,
                            raw_obs=raw_states)  # (1,n,g,g,ch*n_frames)

        # Get mask for the padding
        self.mask = make_mask(alive_agents_ids=self.alive_agents_ids,
                              max_num_agents=self.env.config.max_num_red_agents)  # (1,n)

        # Build policy
        self.policy(self.padded_states, self.mask, training=False)

        # reset episode variables
        self.step = 0

    def initialize_results(self):
        results = {}

        results['episode_rewards'] = []
        results['episode_lens'] = []

        results['alive_red_platoon'] = []
        results['alive_red_company'] = []
        results['alive_reds_ratio'] = []
        results['remaining_red_effective_force_ratio'] = []

        results['initial_red_platoon'] = []
        results['initial_red_company'] = []

        results['alive_blue_platoon'] = []
        results['alive_blue_company'] = []
        results['alive_blues_ratio'] = []
        results['remaining_blue_effective_force_ratio'] = []

        results['initial_blue_platoon'] = []
        results['initial_blue_company'] = []

        results['winner'] = []

        return results

    def test_play(self, current_weights, epsilon):
        # 重みを更新
        self.policy.set_weights(weights=current_weights)

        results = self.initialize_results()

        for _ in range(self.env.config.max_episodes_test_play):
            dones = {}
            dones['all_dones'] = False
            episode_reward = 0

            while not dones['all_dones']:

                q_logits, scores = self.policy(self.padded_states, self.mask, training=False)
                # (1,n,action_dim), [(1,num_heads,n,n),(1,num_heads,n,n)]

                # get alive_agents & all agents actions. action=0 <- do nothing
                actions = {}  # For alive agents

                acts = np.argmax(q_logits[0, :, :], axis=-1)  # (n,)

                for i, a in enumerate(self.alive_agents_ids):
                    agent_id = 'red_' + str(a)

                    if np.random.rand() >= epsilon:  # epsilon-greedy
                        actions[agent_id] = acts[i]
                    else:
                        actions[agent_id] = np.random.randint(low=0,
                                                              high=self.action_space_dim)

                        # One step of Lanchester simulation, for alive agents in env
                next_obserations, rewards, dones, infos = self.env.step(actions)

                # Make next_agents_states, next_agents_adjs, and next_alive_agents_ids,
                # including dummy ones
                next_alive_agents_ids = get_alive_agents_ids(env=self.env)

                ### For alive agents in env
                next_states = {}

                for idx in next_alive_agents_ids:
                    agent_id = 'red_' + str(idx)

                    self.frames[agent_id].append(
                        next_obserations[agent_id]
                    )  # append (g,g,ch) to deque

                    next_states[agent_id] = np.concatenate(
                        self.frames[agent_id], axis=2
                    ).astype(np.float32)  # (g,g,ch*n_frames)

                # Get next_observation list of alive agents
                raw_next_states = []

                for a in next_alive_agents_ids:
                    agent_id = 'red_' + str(a)
                    raw_next_states.append(next_states[agent_id])  # append (g,g,ch*n_frames)

                # Get padded next observations ndarray of all agent
                next_padded_states = \
                    make_padded_obs(
                        max_num_agents=self.env.config.max_num_red_agents,
                        obs_shape=self.obs_shape,
                        raw_obs=raw_next_states
                    )  # (1,n,g,g,ch*n_frames)

                # Get next mask for the padding
                next_mask = \
                    make_mask(
                        alive_agents_ids=next_alive_agents_ids,
                        max_num_agents=self.env.config.max_num_red_agents
                    )  # (1,n)

                # 終了判定
                if self.step > self.env.config.max_steps:

                    for idx in self.alive_agents_ids:
                        agent_id = 'red_' + str(idx)
                        dones[agent_id] = True

                    dones['all_dones'] = True

                # agents_rewards and agents_dones, including dead and dummy ones, for result output
                agents_rewards = []
                agents_dones = []

                for idx in range(self.env.config.max_num_red_agents):

                    if idx in self.alive_agents_ids:
                        agent_id = 'red_' + str(idx)
                        agents_rewards.append(float(rewards[agent_id]))
                        agents_dones.append(dones[agent_id])
                    else:
                        agents_rewards.append(0.0)
                        agents_dones.append(True)

                # Update episode rewards
                episode_reward += np.sum(agents_rewards)

                # Make animation
                if self.env.config.make_animation:
                    self.env.make_animation.add_frame(self.env)  # log-normalized map
                    red = self.env.reds[-1]
                    if red.alive:
                        self.env.make_animation.add_observations_3(next_obserations[red.id])

                        self.env.make_animation_attention_map.add_att_map(
                            relation_kernel=0,  # For relation_kernel 0
                            agent_id=red.id,
                            alive_agents_ids=self.alive_agents_ids,
                            atts=scores,
                        )

                        self.env.make_animation_attention_map.add_att_map(
                            relation_kernel=1,  # For relation_kernel 1
                            agent_id=red.id,
                            alive_agents_ids=self.alive_agents_ids,
                            atts=scores,
                        )

                        self.env.make_animation_attention_map.add_observations_3(
                            next_obserations[red.id]
                        )

                if dones['all_dones']:
                    results['episode_lens'].append(self.step)
                    results['episode_rewards'].append(episode_reward)

                    # Summarize each agent result
                    result_red = summarize_agent_result(self.env.reds)
                    result_blue = summarize_agent_result(self.env.blues)

                    # Decide winner
                    winner = who_wins(result_red, result_blue)

                    # Summarize episode result
                    results = summarize_episode_results(results, result_red, result_blue, winner)

                    # Generate animation
                    if self.env.config.make_animation:
                        self.env.make_animation.generate_movies(self.env)
                        self.env.make_animation_attention_map.generate_movies()

                    # Reset env
                    observations = self.env.reset()
                    self.reset_states(observations)
                else:  # dones['all_done'] ではない時
                    self.alive_agents_ids = next_alive_agents_ids
                    self.padded_states = next_padded_states
                    self.mask = next_mask

                    self.step += 1

        result = summarize_results(results)

        return result


def main():
    """
    Use this to make an animation.  Specify the model
    Be careful params in config, e.g.,  max_episodes_test_play=1,
                                        max_steps, n_frames, key_dim, ...
    """
    from pathlib import Path

    from config import Config

    is_debug = False  # True for debug

    if is_debug:
        print("Debug mode starts. May cause ray memory error.")
    else:
        print("Execution mode starts")

    ray.init(local_mode=is_debug, ignore_reinit_error=True)

    config = Config()
    epsilon = 0.

    # Make dummy_policy and load learned weights
    dummy_policy = MarlTransformerModel(config=config)

    # Build model
    grid_size = config.grid_size
    ch = config.observation_channels
    n_frames = config.n_frames
    max_num_agents = 10

    obs_shape = (grid_size, grid_size, ch * n_frames)

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    raw_obs = []

    for a in alive_agents_ids:
        obs_a = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2])
        raw_obs.append(obs_a)

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape, raw_obs)  # (1,n,g,g,ch*n_frames)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    dummy_policy(padded_obs, mask)

    # Load model
    load_dir = Path(__file__).parent / 'trial-81/models'
    load_name = '/model_83000/'

    dummy_policy.load_weights(str(load_dir) + load_name)

    weights = dummy_policy.get_weights()

    # testerをインスタンス化
    tester = Tester.remote()

    # Start test process
    wip_tester = tester.test_play.remote(current_weights=weights, epsilon=epsilon)

    # Get results
    finished_tester, _ = ray.wait([wip_tester], num_returns=1)

    result = ray.get(finished_tester[0])

    print(f'{config.max_episodes_test_play} test trials:')
    print(f" - mean_episode_rewards = {result['episode_rewards']}")
    print(f" - mean_episode_len = {result['episode_lens']}")

    print(f" - mean_num_alive_reds_ratio = {result['num_alive_reds_ratio']}")
    print(f" - mean_num_alive_red_platoon = {result['num_alive_red_platoon']}"
          f" over mean_num_initial_red_platoon = {result['num_initial_red_platoon']}")
    print(f" - mean_num_alive_red_company = {result['num_alive_red_company']}"
          f" over mean_num_initial_red_company = {result['num_initial_red_company']}")
    print(f" - mean_remaining_red_effective_force_ratio = "
          f"{result['remaining_red_effective_force_ratio']}")

    print(f" - mean_num_alive_blues_ratio = {result['num_alive_blues_ratio']}")
    print(f" - mean_num_alive_blue_platoon = {result['num_alive_blue_platoon']}"
          f" over mean_num_initial_blue_platoon = {result['num_initial_blue_platoon']}")
    print(f" - mean_num_alive_blue_company = {result['num_alive_blue_company']}"
          f" over mean_num_initial_blue_company = {result['num_initial_blue_company']}")
    print(f" - mean_remaining_blue_effective_force_ratio = "
          f"{result['remaining_blue_effective_force_ratio']}")

    print(f" - num_red_win = {result['num_red_win']}")
    print(f" - num_blue_win = {result['num_blue_win']}")
    print(f" - num_draw = {result['draw']}")
    print(f" - num_no_contest = {result['no_contest']}")

    ray.shutdown()


if __name__ == '__main__':
    main()
