import numpy as np
import tensorflow as tf
import os

from config import Config
from sub_models import CNNModel, MultiHeadAttentionModel, QLogitsModel
from utils_transformer import make_mask, make_padded_obs


class MarlTransformerModel(tf.keras.models.Model):
    """
    :inputs: padded obs (None,n,g,g,ch*n_frames),
             mask (None,n,n)
    :return: Q logits (None,n,action_dim), n=max_num_agents
             scores [(None,num_heads,n,n),(None,num_heads,n,n)]

    Model: "marl_transformer"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     input_1 (InputLayer)        [(None, 15, 15, 15, 6)]   0

     cnn_model (CNNModel)        (None, 15, 256)           258944

     dropout (Dropout)           (None, 15, 256)           0

     multi_head_attention_model   ((None, 15, 256),        526080
     (MultiHeadAttentionModel)    (None, 2, 15, 15))

     multi_head_attention_model_  ((None, 15, 256),        526080
     1 (MultiHeadAttentionModel)   (None, 2, 15, 15))

     q_logits_model (QLogitsMode  (None, 15, 5)            395525
     l)

    =================================================================
    Total params: 1,706,629
    Trainable params: 1,706,629
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MarlTransformerModel, self).__init__(**kwargs)

        self.config = config

        """ Prepare sub models """
        self.cnn = CNNModel(config=self.config)

        self.dropout = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.mha1 = MultiHeadAttentionModel(config=self.config)

        self.mha2 = MultiHeadAttentionModel(config=self.config)

        self.q_net = QLogitsModel(config=self.config)

    @tf.function
    def call(self, x, mask, training=True):
        # x: (None,n,g,g,ch*n_frames)=(None,17,20,20,16), mask:(None,n,n)=(none,17,17)

        """ CNN layer """
        features_cnn = self.cnn(x, mask)  # (1,n,hidden_dim)

        """ Dropout layer """
        features_cnn = self.dropout(features_cnn, training=training)

        """ Multi Head Self-Attention layer 1 """
        # features_mha1: (None,n,hidden_dim),
        # score1: (None,num_heads,n,n)
        features_mha1, score1 = self.mha1(features_cnn, mask, training=training)

        """ Multi Head Self-Attention layer 2 """
        # features_mha2: (None,n,hidden_dim),
        # score2: (None,num_heads,n,n)
        features_mha2, score2 = self.mha2(features_mha1, mask, training=training)

        """ Q network (logits output) """
        q_logits = self.q_net(features_mha2, mask, training=training)  # (None,n,action_dim)

        return q_logits, [score1, score2]

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

        features_cnn = self.cnn(x, mask)

        features_cnn = self.dropout(features_cnn, training=True)

        features_mha1, score1 = self.mha1(features_cnn, mask, training=True)
        features_mha2, score2 = self.mha2(features_mha1, mask, training=True)

        q_logits = self.q_net(features_mha2, mask)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=[q_logits, [score1, score2]],
            name='marl_transformer',
        )

        return model


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (grid_size, grid_size, ch * n_frames)

    max_num_agents = config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    raw_obs = []

    for a in alive_agents_ids:
        obs_a = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2])
        raw_obs.append(obs_a)

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape, raw_obs)  # (1,n,g,g,ch*n_frames)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    """ Make model """
    marl_transformer = MarlTransformerModel(config=config)

    q_logits, scores = marl_transformer(padded_obs, mask, training=True)

    """ Summary """
    print('\n')
    print('--------------------------------------- model ---------------------------------------')
    marl_transformer.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        marl_transformer.build_graph(mask),
        to_file=dir_name + '/marl_transformer',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )


if __name__ == '__main__':
    main()
