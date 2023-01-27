"""
rayで並列処理
"""
import json
import os

import ray
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

from config import Config
from actor import Actor
from replay import Replay
from learner import Learner
from tester import Tester


def write_config(config):
    """
    Save Training conditions
    """
    config_list = {
        'max_episodes_test_play': config.max_episodes_test_play,
        'grid_size': config.grid_size,
        'offset': config.offset,

        'action_dim': config.action_dim,
        'observation_channels': config.observation_channels,
        'n_frames': config.n_frames,

        'capacity': config.capacity,
        'compress': config.compress,
        'prioritized_replay': config.prioritized_replay,

        'hidden_dim': config.hidden_dim,
        'key_dim': config.key_dim,
        'num_heads': config.num_heads,
        'dropout_rate': config.dropout_rate,

        'actor_rollout_steps': config.actor_rollout_steps,
        'num_update_cycles': config.num_update_cycles,
        'actor_rollouts_before_train': config.actor_rollouts_before_train,
        'batch_size': config.batch_size,
        'num_minibatchs': config.num_minibatchs,

        'tau': config.tau,
        'gamma': config.gamma,

        'max_steps': config.max_steps,

        'learning_rate': config.learning_rate,
        'loss_coef': config.loss_coef,

        'threshold': config.threshold,
        'mul': config.mul,
        'dt': config.dt,

        'agent_types': config.agent_types,
        'agent_forces': config.agent_forces,

        'red_platoons': config.red_platoons,
        'red_companies': config.red_companies,

        'blue_platoons': config.blue_platoons,
        'blue_companies': config.blue_companies,

        'efficiencies_red': config.efficiencies_red,
        'efficiencies_blue': config.efficiencies_blue,

        'max_num_red_agents': config.max_num_red_agents,
    }

    dir_save = './trial'
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    with open(dir_save + '/training_conditions.json', 'w') as f:
        json.dump(config_list, f, indent=5)


def main(num_actors=8):
    """
    td_errors: (100,1), actor当たり、1回のrolloutは100 stepまで実行
    transitions: [(agents_states, actions, rewards, agents_next_states, dones,
                   adjs, alive_agents_ids)]のリスト, len = 100
    """
    is_debug = False  # True for debug

    if is_debug:
        print("Debug mode starts. May cause ray memory error.")
    else:
        print("Execution mode starts")

    ray.init(local_mode=is_debug, ignore_reinit_error=True)

    logdir = Path(__file__).parent / "log"

    summary_writer = tf.summary.create_file_writer(logdir=str(logdir))

    start = time.time()
    history = []

    config = Config()

    write_config(config)

    # actorのインスタンスをnum_actors個生成
    epsilons = np.linspace(0.01, 0.5, num_actors)
    actors = [Actor.remote(pid=i, epsilon=epsilons[i]) for i in range(num_actors)]

    # learnerをインスタンス化し、define_network()メソッドにより、current_weightsを取得し、ray.put
    learner = Learner.remote()
    current_weights = ray.get(learner.define_network.remote())
    current_weights = ray.put(current_weights)

    # Replay bufferをインスタンス化
    replay = Replay(buffer_size=config.capacity, compress=config.compress)

    # testerをインスタンス化
    tester = Tester.remote()

    # actor.rollout() のobject refsのリストを定義
    wip_actors = [actor.rollout.remote(current_weights) for actor in actors]

    # まず、ある程度の経験を収集するために50回のactor.rollout()の結果を1 rollout分づつ取得し、replayに追加
    for _ in range(config.actor_rollouts_before_train):
        finished_actor, wip_actors = ray.wait(wip_actors, num_returns=1)  # 処理が終了したObjctRefを1つ取得
        td_errors, transitions, pid = ray.get(finished_actor[0])  # ObjectRefから結果を取得
        replay.add(td_errors, transitions)  # Replayに追加
        wip_actors.extend([actors[pid].rollout.remote(current_weights)])  # 新しいobject refsを追加

    # learner.update_networkでネットワーク更新のObjectRefを定義
    # batch_size=self.batch_size（=32）の minibatch を self.num_minibatchs個（=5）生成
    #   minibatch=[sampled_indices, weights, experiences]
    minibatchs = [replay.sample(batch_size=config.batch_size) for _ in range(config.num_minibatchs)]

    wip_learner = learner.update_network.remote(minibatchs=minibatchs)  # network更新  ObjectRef

    minibatchs = [replay.sample(batch_size=config.batch_size) for _ in
                  range(config.num_minibatchs)]  # 次のミニバッチを用意

    # test実施
    wip_tester = tester.test_play.remote(current_weights, epsilon=0.0)

    # while iteration
    update_cycles = config.n0 + 1
    actor_cycles = config.actor_cycles  # Actorが経験をReplayに渡した回数
    test_cycles = update_cycles

    while update_cycles <= config.num_update_cycles:

        # actorのrolloutが終了していれば結果をget
        finished_actor, wip_actors = ray.wait(wip_actors, num_returns=1, timeout=0)

        if finished_actor:
            td_errors, transitions, pid = ray.get(finished_actor[0])

            # 結果をreplayに追加
            replay.add(td_errors, transitions)

            # 新しいactor.rollout()のObjectRefを追加
            wip_actors.extend([actors[pid].rollout.remote(current_weights)])

            actor_cycles += 1

        # Learnerのタスク完了結果を取得。未終了時は、finished_learner=[]が返る
        finished_learner, _ = ray.wait([wip_learner], timeout=0)

        if finished_learner:
            # current weightをlearner.update_network()から取得
            current_weights, indices, td_errors, mean_loss = \
                ray.get(finished_learner[0])

            # print(f'mean_loss={mean_loss}')

            # 新しいupdate_networkのObjectRefを追加
            wip_learner = learner.update_network.remote(minibatchs=minibatchs)

            # current_weightをray.put
            current_weights = ray.put(current_weights)

            # replay bufferの優先度更新；replay.update_priority
            replay.update_priprity(indices, td_errors)

            # 次のminibatch setを用意
            minibatchs = [replay.sample(batch_size=config.batch_size)
                          for _ in range(config.num_minibatchs)]

            update_cycles += 1

            with summary_writer.as_default():
                tf.summary.scalar("mean_loss of training", mean_loss, step=update_cycles)
                tf.summary.scalar("actor_cycles", actor_cycles, step=update_cycles)

        # Test process
        finished_tester, _ = ray.wait([wip_tester], timeout=0)

        if finished_tester:
            result = ray.get(finished_tester[0])
            print(f"test_cycles={test_cycles}, test_score={result['episode_rewards']}, "
                  f"episode_len={result['episode_lens']}")
            history.append((test_cycles, result['episode_rewards']))

            with summary_writer.as_default():
                tf.summary.scalar(
                    "mean_episode_reward of tests", result['episode_rewards'], step=test_cycles)
                tf.summary.scalar(
                    "mean_episode_len of tests", result['episode_lens'], step=test_cycles)

                tf.summary.scalar(
                    "mean_num_alive_red_ratio", result['num_alive_reds_ratio'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_red_platoon",
                    result['num_alive_red_platoon'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_red_company",
                    result['num_alive_red_company'], step=test_cycles)
                tf.summary.scalar(
                    "mean_remaining_red_effective_force_ratio",
                    result['remaining_red_effective_force_ratio'], step=test_cycles)

                tf.summary.scalar(
                    "mean_num_alive_blue_ratio", result['num_alive_blues_ratio'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_blue_platoon",
                    result['num_alive_blue_platoon'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_blue_company",
                    result['num_alive_blue_company'], step=test_cycles)
                tf.summary.scalar(
                    "mean_remaining_blue_effective_force_ratio",
                    result['remaining_blue_effective_force_ratio'], step=test_cycles)

                tf.summary.scalar(
                    "num_red_win", result['num_red_win'], step=test_cycles)
                tf.summary.scalar(
                    "num_blue_win", result['num_blue_win'], step=test_cycles)
                tf.summary.scalar(
                    "num_draw", result['draw'], step=test_cycles)
                tf.summary.scalar(
                    "num_no_contest", result['no_contest'], step=test_cycles)

            # 次のプロセスを準備
            test_cycles = update_cycles

            wip_tester = tester.test_play.remote(current_weights, epsilon=0.0)

    # 表示処理
    wallclocktime = round(time.time() - start, 2)
    cycles, scores = zip(*history)  # historyの中身を取り出す

    plt.plot(cycles, scores)
    plt.title(f'total time: {wallclocktime} sec')
    plt.ylabel('test_score(epsilon=0.01')
    plt.savefig('history.png')

    ray.shutdown()


if __name__ == '__main__':
    main(num_actors=4)  # default=4 for GCP
