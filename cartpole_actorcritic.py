import numpy as np
import tensorflow as tf
import gym
from matplotlib import pyplot as plt

env = gym.make('CartPole-v1')

observation_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = 32
discount = 0.95
lr = 0.001
iterations = 1000

class PGAgent:

    def __init__(self):
        self.inputs = tf.placeholder(shape=[None,observation_size], dtype=tf.float32)
        init = tf.contrib.layers.xavier_initializer()

        #Policy gradient network
        self.W1 = tf.Variable(init([observation_size,hidden_size]))
        self.dense = tf.nn.relu(tf.matmul(self.inputs, self.W1))
        self.W2 = tf.Variable(init([hidden_size,action_size]))
        self.dense2 = tf.matmul(self.dense,self.W2)
        self.predict = tf.nn.softmax(self.dense2)

        #Value network
        self.W3 = tf.Variable(init([observation_size,hidden_size]))
        self.v1 = tf.nn.relu(tf.matmul(self.inputs, self.W3)) #dense layer
        self.W4 = tf.Variable(init([hidden_size,1]))
        self.value = tf.matmul(self.v1,self.W4)

        self.actions = tf.placeholder(shape=[None],dtype = tf.int32)
        self.targets = tf.placeholder(shape=[None],dtype = tf.float32)

        self.logits = self.dense2
        self.negative_likelihoods = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.transpose(self.actions), logits=self.logits)
        self.weighted_negative_likelihoods = -tf.multiply(self.negative_likelihoods, self.targets)
        self.pg_error = tf.reduce_mean(self.weighted_negative_likelihoods)
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
            probs = np.reshape(sess.run(network.predict, feed_dict={network.inputs:preprocess(s,1)}),action_size)
            a = np.random.choice(range(action_size), p=probs)
            ns, r, d, _ = env.step(a)
            episode_reward += r
            #Update value network
            target = r + discount * sess.run(network.value, feed_dict={network.inputs:preprocess(ns,1)})
            _ = sess.run(network.step2, feed_dict={network.inputs:preprocess(s,1),network.targets:np.reshape(target, [1])})
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
