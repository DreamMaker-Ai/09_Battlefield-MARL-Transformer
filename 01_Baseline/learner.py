from pathlib import Path

import numpy as np
import ray
import tensorflow as tf

from battlefield_strategy import BattleFieldStrategy
from models import MarlTransformerModel
from utils_transformer import make_mask, make_padded_obs, make_next_states_for_q


# @ray.remote
@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self):
        self.env = BattleFieldStrategy()

        self.action_space_dim = self.env.action_space.n
        self.gamma = self.env.config.gamma

        self.q_network = MarlTransformerModel(config=self.env.config)

        self.target_q_network = MarlTransformerModel(config=self.env.config)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.env.config.learning_rate)

        self.count = self.env.config.n0 + 1

    def define_network(self):
        """
        Q-network, Target_networkを定義し、current weightsを返す
        """
        # self.q_network.compile(optimizer=self.optimizer, loss='mse')

        # Build graph with dummy inputs
        grid_size = self.env.config.grid_size
        ch = self.env.config.observation_channels
        n_frames = self.env.config.n_frames

        obs_shape = (grid_size, grid_size, ch * n_frames)

        max_num_agents = self.env.config.max_num_red_agents

        alive_agents_ids = [0, 2]
        raw_obs = []

        for a in alive_agents_ids:
            obs_a = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2])
            raw_obs.append(obs_a)

        # Get padded_obs and mask
        padded_obs = make_padded_obs(max_num_agents, obs_shape, raw_obs)  # (1,n,g,g,ch*n_frames)

        mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

        self.q_network(padded_obs, mask, training=True)
        self.target_q_network(padded_obs, mask, training=False)

        # Load weights
        if self.env.config.model_dir:
            self.q_network.load_weights(self.env.config.model_dir)

        # Q networkのCurrent weightsをget
        current_weights = self.q_network.get_weights()

        # Q networkの重みをTarget networkにコピー
        self.target_q_network.set_weights(current_weights)

        return current_weights

    def update_network(self, minibatchs):
        """
        minicatchsを使ってnetworkを更新
        minibatchs = [minibatch,...], len=16(default)

        minibatch = [sampled_indices, correction_weights, experiences], batch_size=32(default)
            - sampled_indices: [int,...], len = batch_size
            - correction_weights: Importance samplingの補正用重み, (batch_size,), ndarray
            - experiences: [experience,...], len=batch_size
                experience =
                    (
                        (padded_)states,  # (1,n,g,g,ch*n_frames)
                        (padded_)actions,  # (1,n)
                        (padded_)rewards,  # (1,n)
                        next_(padded_)states,  # (1,n,g,g,ch*n_frames)
                        (padded_)dones,  # (1,n), bool
                        masks,  # (1,n), bool
                        next_masks,  # (1,n), bool
                        next_(padded_)states_for_q,  # (1,n,g,g,ch*n_frames)
                        alive_agents_ids,  # (1,a), a=num_alive_agent, object
                    )

                ※ experience.states等で読み出し

        :return:
            current_weights: 最新のnetwork weights
            indices_all: ミニバッチに含まれるデータのインデクス, [int,...], len=batch*16(default)
            td_errors_all: ミニバッチに含まれるデータのTD error, [(batch,n),...], len=16(default)
        """
        indices_all = []
        td_errors_all = []
        losses = []

        for (indices, correction_weights, experiences) in minibatchs:
            # indices:list, len=32; correction_weights: ndarray, (32,1); experiences: lsit, len=32
            # experiencesをnetworkに入力するshapeに変換

            # process in minibatch
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            masks = []
            next_masks = []
            next_states_for_q = []
            alive_agents_ids = []

            for i in range(len(experiences)):
                states.append(experiences[i].states)
                actions.append(experiences[i].actions)
                rewards.append(experiences[i].rewards)
                next_states.append(experiences[i].next_states)
                dones.append(experiences[i].dones)
                masks.append(experiences[i].masks)
                next_masks.append(experiences[i].next_masks)
                next_states_for_q.append(experiences[i].next_states_for_q)
                alive_agents_ids.append(experiences[i].alive_agents_ids)

            # list -> ndarray
            states = np.vstack(states)  # (batch,n,g,g,ch*n_frames)=(32,17,20,20,16)
            actions = np.vstack(actions)  # (batch,n)=(32,17)
            rewards = np.vstack(rewards)  # (32,17)
            next_states = np.vstack(next_states)  # (32,17,20,20,16)
            dones = np.vstack(dones)  # (32,17), bool
            masks = np.vstack(masks)  # (32,17), bool
            next_masks = np.vstack(next_masks)  # (32,17), bool
            next_states_for_q = np.vstack(next_states_for_q)  # (32,17,20,20,16)

            # ndarray -> tf.Tensor
            states = tf.convert_to_tensor(states, dtype=tf.float32)  # (32,17,20,20,16)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # (32,17)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)  # (32,17)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)  # (32,17,20,20,16)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # (32,17), bool->float32
            masks = tf.convert_to_tensor(masks, dtype=tf.float32)  # (32,17), bool->float32
            next_masks = \
                tf.convert_to_tensor(next_masks, dtype=tf.float32)  # (32,17), bool->float32
            next_states_for_q = \
                tf.convert_to_tensor(next_states_for_q, dtype=tf.float32)  # (32,17,20,20,16)

            # Target valueの計算
            next_q_logits, _ = \
                self.target_q_network(next_states_for_q, masks, training=False)  # (32,17,5)
            next_actions = tf.argmax(next_q_logits, axis=-1)  # (32,17)
            next_actions = tf.cast(next_actions, dtype=tf.int32)
            next_actions_one_hot = \
                tf.one_hot(next_actions, depth=self.action_space_dim)  # (32,17,5)

            next_maxQ = next_q_logits * next_actions_one_hot  # (32,17,5)
            next_maxQ = tf.reduce_sum(next_maxQ, axis=-1)  # (32,17)

            TQ = rewards + self.gamma * (1 - dones) * next_maxQ  # (32,17)

            # ロス計算
            with tf.GradientTape() as tape:
                q_logits, _ = self.q_network(states, masks, training=True)  # (32,17,5)
                actions_one_hot = tf.one_hot(actions, depth=self.action_space_dim)  # (32,17,5)

                Q = q_logits * actions_one_hot  # (32,17,5)
                Q = tf.reduce_sum(Q, axis=-1)  # (32,17)

                td_errors = tf.square(TQ - Q)  # (32,17)

                masked_td_errors = td_errors * masks  # (32,17)
                masked_td_errors = \
                    tf.reduce_sum(masked_td_errors, axis=-1) / \
                    tf.reduce_sum(masks, axis=-1)  # (32,)

                loss = tf.reduce_mean(correction_weights * masked_td_errors) * \
                       self.env.config.loss_coef

                losses.append(loss.numpy())

            # 勾配計算と更新
            grads = tape.gradient(loss, self.q_network.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 40)

            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

            # Compute priority update
            if self.env.config.prioritized_replay:
                priority_td_errors = np.abs((TQ - Q).numpy())  # (32,17)
                masked_priority_td_errors = priority_td_errors * masks.numpy()  # (32,17)
                masked_priority_td_errors = \
                    np.sum(masked_priority_td_errors, axis=-1) / \
                    np.sum(masks.numpy(), axis=-1)  # (32,)

            else:
                masked_priority_td_errors = np.ones((self.env.config.batch_size,),
                                                    dtype=np.float32)  # (32,)

            # learnerの学習に使用した経験のインデクスとTD-errorのリスト
            indices_all += indices
            td_errors_all += masked_priority_td_errors.tolist()  # len=32のリストに変換

        # 最新のネットワークweightsをget
        current_weights = self.q_network.get_weights()

        # Target networkのweights更新: Soft update
        target_weights = self.target_q_network.get_weights()

        for w in range(len(target_weights)):
            target_weights[w] = \
                self.env.config.tau * current_weights[w] + \
                (1. - self.env.config.tau) * target_weights[w]

        self.target_q_network.set_weights(target_weights)

        # Save model
        if self.count % 100 == 0:
            save_dir = Path(__file__).parent / 'models'
            save_name = '/model_' + str(self.count) + '/'

            self.q_network.save_weights(str(save_dir) + save_name)

        self.count += 1

        return current_weights, indices_all, td_errors_all, np.mean(loss)
