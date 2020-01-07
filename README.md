# Actor-Critic-for-the-Pole-Balancing-Problem
Actor-Critic Policy Gradient implementation for the pole balancing problem using TensorFlow and OpenAI Gym. Note that this implementation is not A3C as it does not use multithreading, it simply uses separate value and policy networks. Included are a regular Actor-Critic agent and an agent using N-Step Actor-Critic. Both perform 1000 trials and produce a graph of their performance over time. The standard Actor-Critic agent tends to outperform the N-Step Actor Critic agent in this environment. 

Additionally, an implementation of Actor-Critic is shown for continuous action spaces, namely the Pendulum-v0 environment. While not mastering the environment, there is a definitive trend in scores that suggests the agent is learning. It outperforms a vanilla policy gradient algorithm I included for continuous action spaces which fails to learn the Pendulum environment. Perhaps more changes could be made to optimize it.

These agents were written in May 2018.
