# 09_Battlefield-MARL-Transformer

Implementation of the following blog post

[Preliminary Implementation of MARL-Transformer for Generating Battlefield Strategy (1)](https://medium.com/@tym406/preliminary-implementation-of-marl-transformer-for-generating-battlefield-strategy-1-b3ae18631082?source=your_stories_page-------------------------------------)

[Preliminary Implementation of MARL-Transformer for Generating Battlefield Strategy (2)](https://medium.com/@tym406/preliminary-implementation-of-marl-transformer-for-generating-battlefield-strategy-2-f32bb5187917?source=your_stories_page-------------------------------------)


## Abstract
In this preliminary implementation, I propose the MARL-Transformer, which applies the transformer used in natural language processing to multi-agent reinforcement learning (MARL).

By using the MARL-Transformer, the followings are possible.
(1) An arbitrary number of agents can be processed during testing, which is different from the number of agents during training. The number of agents during testing can be arbitrarily larger than during training.
(2) The number of agents can be increased or decreased in the course of an episode.
(3) Communication networks can be incorporated as an attention mask.
(4) The content of communication between agents can be shaped to be appropriate through learning. No handcrafting is necessary.

In the application example, MARL-Transformer is subjected to centralized training and centralized test execution to generate battlefield strategies for future application to the military C2 (Command and Control) system.

The multi-agent environment is a grid battlefield consisting of multiple platoons and companies for each of the red and blue armies. Each of these platoons or companies is an agent. Each agent consists of a swarm of unmanned systems. The number of unmanned systems for each agent is attrition over the course of the battle according to the Lanchester model.

The learning objective is to generate a battlefield strategy for the red army that successfully moves the red agents initially deployed in random positions on the battlefield to minimize the attrition of the red army as much as possible, while increasing the red army’s win rate as much as possible.

For training, the average number of initial unmanned systems for the red and blue armies shall be equal. Since this is a preliminary implementation, all blue agents are assumed to be stationary so that the problem becomes a Markov Decision Process (MDP).

In the test, applying the MARL-Transformer, the red army achieves a win rate of about 89% and a loss rate of about 0.1%, (i.e., no-contest is about 11%).

The flexibility of the MARL-Transformer and the robustness of the learned red army strategy will be tested by increasing the number of agents in the red and blue armies from those in the training. Even if the number of agents is increased several times, MARL-Transformer can still apply the learned strategies and ensure a certain win rate.

Furthermore, when additional red agents are deployed as reinforcements in the middle of an episode, the MARL-Transformer can ensure a red army win by extending the strategy to the reinforcements as well.

Layer normalization, batch size, number of stacked transformer blocks, number of frame stacks, and impact of obstacles will also be investigated.
