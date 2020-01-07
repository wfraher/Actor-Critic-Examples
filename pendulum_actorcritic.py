import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from matplotlib import pyplot as plt

env = gym.make('Pendulum-v0')

observation_size = env.observation_space.shape[0]
action_size = env.action_space.low.shape[0]
hidden_size = 8
discount = 0.99
lr = 0.001
iterations = 1000

class PGAgent:

    def __init__(self):
        self.inputs = tf.placeholder(shape=[None,observation_size], dtype=tf.float32)

        #Policy gradient network
        self.pg_net = slim.fully_connected(self.inputs, hidden_size, activation_fn = tf.nn.relu)
        self.pg_net = slim.fully_connected(self.pg_net, hidden_size, activation_fn = tf.nn.relu)
        self.pg_net = slim.fully_connected(self.pg_net, hidden_size, activation_fn = tf.nn.relu)
        self.mean = slim.fully_connected(self.pg_net, action_size, activation_fn = None)
        self.standard = slim.fully_connected(self.pg_net, action_size, activation_fn = None)
        self.distribution = tf.distributions.Normal(self.mean, self.standard)
        self.action = self.distribution.sample([1])

        #Value network
        self.dense = slim.fully_connected(self.inputs, hidden_size, activation_fn = tf.nn.relu)
        self.dense = slim.fully_connected(self.dense, hidden_size, activation_fn = tf.nn.relu)
        self.dense = slim.fully_connected(self.dense, hidden_size, activation_fn = tf.nn.relu)        
        self.value = slim.fully_connected(self.dense, action_size, activation_fn = None)

        self.actions = tf.placeholder(shape=[None],dtype = tf.float32)
        self.targets = tf.placeholder(shape=[None],dtype = tf.float32)

        self.logprobs = self.distribution.log_prob(self.actions)
        self.pg_error = tf.reduce_mean(-tf.multiply(self.logprobs,self.targets))
        self.value_error = tf.reduce_mean(tf.square(self.targets - self.value))

        self.opt = tf.train.AdamOptimizer(lr)
        self.step = self.opt.minimize(self.pg_error)
        self.step2 = self.opt.minimize(self.value_error)

def preprocess(states, size):
    return np.reshape(states, [size, observation_size])

def compute_discounted_R(R, discount_rate=discount):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):

        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean()
    discounted_r /= discounted_r.std()

    return discounted_r

tf.reset_default_graph()
network = PGAgent()
total_rewards = []

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        s = env.reset()
        d = False
        episode_reward = 0
        
        while not d:
            a = sess.run(network.action, feed_dict={network.inputs:preprocess(s,1)})
            ns, r, d, _ = env.step(a)
            r = np.asscalar(r)
            episode_reward += r
            #Update value network
            target = r + discount * sess.run(network.value, feed_dict={network.inputs:preprocess(ns,1)})
            _ = sess.run(network.step2, feed_dict={network.inputs:preprocess(s,1),network.targets:np.reshape(target, [1])})

            #Update policy network
            advantage = (r + discount * sess.run(network.value, feed_dict={network.inputs:preprocess(ns,1)}) - sess.run(network.value, feed_dict={network.inputs:preprocess(s,1)}))
            _ = sess.run(network.step, feed_dict={network.inputs:preprocess(s,1), network.actions:np.reshape(a,[1]), network.targets:np.hstack(advantage)})
            s = ns
            if d:
                break
        
        print 'Episode ' + str(i) + ' finished with reward ' + str(episode_reward)
        total_rewards.append(episode_reward)

plt.plot(total_rewards)
print np.mean(total_rewards)
plt.show()
