import tensorflow as tf
import numpy as np
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

np.random.seed(1)
tf.compat.v1.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]  # you can try different target replacement strategies
# MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = False
ENV_NAME = 'Pendulum-v0'


###############################  Actor  ####################################


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement, S, S_):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.S = S
        self.S_ = S_

        with tf.compat.v1.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(self.S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(self.S_, scope='target_net', trainable=False)

        self.e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.compat.v1.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.compat.v1.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 64, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={self.S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        action = np.argmax(self.sess.run(self.a, feed_dict={self.S: s})[0])
        return action  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.compat.v1.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_, S, S_, R):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement
        self.S = S
        self.S_ = S_
        self.R = R

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)  # stop critic update flows to actor
            self.q = self._build_net(self.S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(self.S_, a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                        scope='Critic/eval_net')
            self.t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                        scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = self.R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]  # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={self.S: s, self.a: a, self.R: r, self.S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


class DeepDeterministicPolicyGradient():
    def __init__(self, action_dim, state_dim, action_bound, LR_A, LR_C, REPLACEMENT, GAMMA, EPSILON=0.1,
                 MEMORY_CAPACITY=10000, OUTPUT_GRAPH=False, BATCH_SIZE=32):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.REPLACEMENT = REPLACEMENT
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.EPSILON = EPSILON

        tf.compat.v1.reset_default_graph()

        # all placeholder for tf
        with tf.name_scope('S'):
            S = tf.compat.v1.placeholder(tf.float32, shape=[None, state_dim], name='s')
        with tf.name_scope('R'):
            R = tf.compat.v1.placeholder(tf.float32, [None, 1], name='r')
        with tf.name_scope('S_'):
            S_ = tf.compat.v1.placeholder(tf.float32, shape=[None, state_dim], name='s_')

        self.sess = tf.compat.v1.Session()
        self.actor = Actor(self.sess, self.action_dim, self.action_bound, self.LR_A, self.REPLACEMENT, S, S_)
        self.critic = Critic(self.sess, self.state_dim, self.action_dim, self.LR_C, self.GAMMA, self.REPLACEMENT,
                             self.actor.a, self.actor.a_, S, S_, R)
        self.actor.add_grad_to_graph(self.critic.a_grads)
        self.sess.run(tf.global_variables_initializer())

        self.M = Memory(MEMORY_CAPACITY, dims=2 * self.state_dim + 2)

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, observation):
        if np.random.uniform() < self.EPSILON:
            action = np.random.randint(0, self.action_dim)
        else:
            action = self.actor.choose_action(observation)
        return action

    def store_transition(self, s, a, r, s_):
        self.M.store_transition(s, a, r / 10, s_)

    def learn(self):
        b_M = self.M.sample(self.BATCH_SIZE)
        b_s = b_M[:, :self.state_dim]
        b_a = b_M[:, self.state_dim: self.state_dim + self.action_dim]
        b_r = b_M[:, -self.state_dim - 1: -self.state_dim]
        b_s_ = b_M[:, -self.state_dim:]

        self.critic.learn(b_s, b_a, b_r, b_s_)
        self.actor.learn(b_s)
