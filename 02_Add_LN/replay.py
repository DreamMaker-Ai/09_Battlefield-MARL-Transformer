import zlib
from dataclasses import dataclass
import numpy as np
import random
import pickle


@dataclass
class Experience:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    masks: np.ndarray
    next_masks: np.ndarray
    next_states_for_q: np.ndarray
    alive_agents_ids: np.ndarray


class Replay:
    def __init__(self, buffer_size, compress=True):
        self.buffer_size = buffer_size
        self.priorities = SumTree(capacity=self.buffer_size)  # Priority(|TD_error|)の入れ物
        self.buffer = [None] * self.buffer_size

        # 優先度計算に使うパラメータ
        self.alpha = 0.6
        self.beta = 0.4

        self.count = 0  # 現在のbuffer index
        self.is_full = False  # Bufferが満杯か否か

        self.compress = compress  # 圧縮するか否か

    def add(self, td_errors, transitions):
        """
        actorの経験（td_erros（優先度）、transitions）を追加

        td_errors: (10,)
        transitions=[transition,...], len=10
            transition =
                (
                    self.padded_states,  # (1,n,g,g,ch*n_frames)
                    padded_actions,  # (1,n)
                    padded_rewards,  # (1,n)
                    next_padded_states,  # (1,n,g,g,ch*n_frames)
                    padded_dones,  # (1,n), bool
                    self.mask,  # (1,n), bool
                    next_mask,  # (1,n), bool
                    next_padded_states_for_q,  # (1,n,g,g,ch*n_frames)
                    alive_agents_ids,  # (1,a), a=num_alive_agent, object
                )
        """
        assert len(td_errors) == len(transitions)

        priorities = (np.abs(td_errors) + 1e-5) ** self.alpha  # (10,)

        for priority, transition in zip(priorities, transitions):
            self.priorities[self.count] = priority  # int

            exp = Experience(*transition)

            if self.compress:
                exp = zlib.compress(pickle.dumps(exp))

            self.buffer[self.count] = exp
            self.count += 1

            if self.count == self.buffer_size:
                self.count = 0
                self.is_full = True

    def update_priprity(self, indices, td_errors):
        """
        Learnerの学習後に、Learnerが使った経験の優先度（td_errors）を更新
            indices=[int,...], len=batch*num_minibatchs=32*10=320, (indices_all in 'learner')
            td_errors=[float,...], len=batch*num_minibatchs=32*10=320, (td_errors_all in 'learner')
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5) ** self.alpha  # (n,)
            self.priorities[idx] = priority

    def sample(self, batch_size):
        """
        batch_sizeのbatchを、優先度に従って用意
        :return:
            sampled_indices: サンプルした経験のインデクスのリスト, [int,...], len=batch_size
            weights: サンプルした経験の重み(Importance samplingの補正用の重み)のリスト,
                     ndarray, (batch_size,)
            experience: サンプルした経験、transitionのリスト, [transition,...], len=batch_size
        """
        sampled_indices = [self.priorities.sample() for _ in range(batch_size)]

        # compute prioritized experience replay weights (Importance samplingの補正重み）
        weights = []
        current_replay_size = len(self.buffer) if self.is_full else self.count
        for idx in sampled_indices:
            probability = self.priorities[idx] / self.priorities.sum()
            weight = (current_replay_size * probability) ** (-self.beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)  # 安定性の理由から正規化, np.array, (32,)

        if self.compress:
            experiences = [pickle.loads(zlib.decompress(self.buffer[idx]))
                           for idx in sampled_indices]
        else:
            experiences = [self.buffer[idx] for idx in sampled_indices]

        return sampled_indices, weights, experiences  # weights: 補正用重み


class SumTree:
    """
    Sum-tree implementation
    https://colab.research.google.com/drive/1yYXAg5M9BoDPzEh6iS6LUB4P4S-2n3e9?usp=sharing
    """

    def __init__(self, capacity: int):
        self.capacity = capacity  # capacity=N, データ数の最大値
        self.values = [0 for _ in range(2 * capacity)]  # 2N-1個の入れ物を用意

    def __str__(self):
        # print(sumtree)で、データの重みを表示
        return str(self.values[self.capacity:])

    def __setitem__(self, idx, val):
        """
        1. sumtree[idx]=valにより、データの重みを葉に格納
        2. 葉から遡って親ノードの値をroot nodeまで計算
        """
        idx = idx + self.capacity  # データ部分のインデクス
        self.values[idx] = val  # データの重みを格納

        # root nodeまで遡って親に子の合計値を計算していく
        current_idx = idx // 2  # 親のインデクス
        while current_idx >= 1:  # root nodeまで遡って計算
            idx_lchild = current_idx * 2  # 左の子のインデクス
            idx_rchild = current_idx * 2 + 1  # 右の子のインデクス
            self.values[current_idx] = self.values[idx_lchild] + self.values[idx_rchild]
            current_idx = current_idx // 2  # 1つノードをさかのぼる

    def __getitem__(self, idx):
        """
        sumtree[idx]でvalを返す
        """
        idx = idx + self.capacity  # データ部分のインデクス
        return self.values[idx]

    def sum(self):
        """
        root nodeの値（=葉の合計値）を返す
        """
        return self.values[1]

    def sample(self, z=None):
        """
        データの重みに比例した割合で、データのインデクスをサンプリングする
        """
        z = random.uniform(0, self.sum()) if z is None else z

        current_idx = 1  # root node
        while current_idx < self.capacity:
            idx_lchild = 2 * current_idx  # 左の子のインデクス
            idx_rchild = 2 * current_idx + 1  # 右の子のインデクス

            # 左の子の値よりもzが大きい時は右の子ノードへ移り、zを更新
            if self.values[idx_lchild] < z:
                current_idx = idx_rchild
                z = z - self.values[idx_lchild]
            else:
                current_idx = idx_lchild

        # 見かけ上のインデクス（データのインデクス）に戻して返す
        return current_idx - self.capacity
