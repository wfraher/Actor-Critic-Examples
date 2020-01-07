import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from matplotlib import pyplot as plt

env = gym.make('Pendulum-v0')

observation_size = env.observation_space.shape[0]
action_size = env.action_space.low.shape[0]
hidden_size = 8
lr = 0.005
iterations = 1000

class PGAgent:

    def __init__(self):
        self.inputs = tf.placeholder(shape=[None,observation_size], dtype=tf.float32)
        self.pg_net = slim.repeat(self.inputs, 2, slim.fully_connected, hidden_size, activation_fn=tf.nn.relu)
        self.mean = slim.fully_connected(self.pg_net, action_size, activation_fn=None)  #the mean of the distribution, which we parameterize.
        self.standard = slim.fully_connected(self.pg_net, action_size, activation_fn=None) #standard deviation. We parameterize it in this case.
        self.distribution = tf.distributions.Normal(self.mean, self.standard)
        self.action = self.distribution.sample([1])

        self.actions = tf.placeholder(shape=[None],dtype = tf.float32)
        self.rewards = tf.placeholder(shape=[None],dtype = tf.float32)

        self.logprobs = self.distribution.log_prob(self.actions)
        self.loss = tf.reduce_mean(tf.multiply(self.logprobs, self.rewards))

        self.opt = tf.train.AdamOptimizer(lr)
        self.step = self.opt.minimize(self.loss)

def preprocess(states, size):
    return np.reshape(states, [size, observation_size])

def compute_discounted_R(R, discount_rate=.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):

        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean()
    discounted_r /= discounted_r.std()

    return np.squeeze(discounted_r) #could be moved to the update step for np.squeeze

tf.reset_default_graph()
network = PGAgent()
total_rewards = []

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        s = env.reset()
        d = False
        episode_reward = 0
        states = []
        actions = []
        rewards = []
        grads = []

        while not d:
            a = sess.run(network.action, feed_dict={network.inputs:preprocess(s,1)})
            ns, r, d, _ = env.step(a)
            episode_reward += np.asscalar(r)
            states.append(preprocess(s,1))
            actions.append(a)
            rewards.append(r)
            s = ns
            if d:
                break
                
        _ = sess.run(network.step, feed_dict={network.inputs:np.vstack(states), network.actions:np.squeeze(actions), network.rewards:compute_discounted_R(rewards)})

        print 'Episode ' + str(i) + ' finished with reward ' + str(episode_reward)
        total_rewards.append(episode_reward)

plt.plot(total_rewards)
print np.mean(total_rewards)
plt.show()
