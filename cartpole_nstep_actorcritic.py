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
        with tf.variable_scope("Policy"):
            self.W1 = tf.Variable(init([observation_size,hidden_size]))
            self.dense = tf.nn.relu(tf.matmul(self.inputs, self.W1))
            self.W2 = tf.Variable(init([hidden_size,action_size]))
            self.dense2 = tf.matmul(self.dense,self.W2)
            self.predict = tf.nn.softmax(self.dense2)

        #Value network
        with tf.variable_scope("Value"):
            self.W3 = tf.Variable(init([observation_size,hidden_size]))
            self.v1 = tf.nn.relu(tf.matmul(self.inputs, self.W3)) #dense layer
            self.W4 = tf.Variable(init([hidden_size,1]))
            self.value = tf.matmul(self.v1,self.W4)

        self.actions = tf.placeholder(shape=[None],dtype = tf.int32)
        self.targets = tf.placeholder(shape=[None],dtype = tf.float32)

        #Gradient placeholders, we need this as for n-step variations we compute running gradients
        policyvariables = tf.trainable_variables(scope="Policy")
        self.policygradients = []
        for variable in policyvariables:
            gradient = tf.placeholder(tf.float32)
            self.policygradients.append(gradient)

        valuevariables = tf.trainable_variables(scope="Value")
        self.valuegradients = []
        for variable in valuevariables:
            gradient = tf.placeholder(tf.float32)
            self.valuegradients.append(gradient)

        self.entropy = - tf.reduce_sum(self.predict * tf.log(self.predict))

        self.logits = self.dense2
        self.negative_likelihoods = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.transpose(self.actions), logits=self.logits)
        self.weighted_negative_likelihoods = -tf.multiply(self.negative_likelihoods, self.targets-self.value)
        self.pg_error = tf.gradients(tf.reduce_mean(self.weighted_negative_likelihoods) + self.entropy, tf.trainable_variables(scope="Policy"))
        self.value_error = tf.gradients(tf.reduce_mean(tf.square(self.targets - self.value)), tf.trainable_variables(scope="Value"))

        self.opt = tf.train.AdamOptimizer(lr)
        self.step = self.opt.apply_gradients(zip(self.policygradients, tf.trainable_variables(scope="Policy")))
        self.step2 = self.opt.apply_gradients(zip(self.valuegradients, tf.trainable_variables(scope="Value")))

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
    gradtheta = sess.run(tf.trainable_variables(scope="Policy"))
    gradvalue = sess.run(tf.trainable_variables(scope="Value"))
    for idx, gradient in enumerate(gradtheta):
        gradtheta[idx] = gradient * 0
    for idx, gradient in enumerate(gradvalue):
        gradvalue[idx] = gradient * 0

    for i in range(iterations):
        s = env.reset()
        d = False
        states = []
        rewards = []
        actions = []
        episode_reward = 0
        qvals = []
        
        while not d:
            probs = np.reshape(sess.run(network.predict, feed_dict={network.inputs:preprocess(s,1)}),action_size)
            a = np.random.choice(range(action_size), p=probs)
            ns, r, d, _ = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            episode_reward += r
            s = ns
            if d:
                break
        R = 0
        for idx, state in enumerate(reversed(states)):
            R = rewards[idx] + discount * R
            #or you could like store gradients in a list
            gtheta = sess.run(network.pg_error, feed_dict={network.inputs:preprocess(state,1), network.actions:np.reshape(actions[idx],[1]), network.targets:np.reshape(R,[1])})
            for index, gradient in enumerate(gtheta):
                gradtheta[index] += gradient
            gvalue = sess.run(network.value_error, feed_dict={network.inputs:preprocess(state,1), network.targets:np.reshape(R,[1])})
            for index, gradient in enumerate(gvalue):
                gradvalue[index] += gradient
        _ = sess.run(network.step, feed_dict=dict(zip(network.policygradients, gradtheta)))
        _2 = sess.run(network.step2, feed_dict=dict(zip(network.valuegradients, gradvalue)))
        for idx, gradient in enumerate(gradtheta):
            gradtheta[idx] = gradient * 0
        for idx, gradient in enumerate(gradvalue):
            gradvalue[idx] = gradient * 0

        print 'Episode ' + str(i) + ' finished with reward ' + str(episode_reward)
        total_rewards.append(episode_reward)

plt.plot(total_rewards)
print np.mean(total_rewards)
plt.show()
