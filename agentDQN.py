import numpy as np
import tensorflow as tf
import time


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            sess,
            name,
            learning_rate=0.001,
            reward_decay=0.1,
            replace_target_iter=300,
            memory_size=5000,
            batch_size=128,
    ):
        self.featureslen = 14
        self.n_actions = n_actions
        self.nplace = int(n_features // self.featureslen)
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.005

        # total learning step
        self.learn_step_counter = 0
        self.sess = sess
        self.name = name

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        with tf.variable_scope(self.name):
            # consist of [target_net, evaluate_net]
            self._build_net()

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/eval_net')

            with tf.variable_scope('hard_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            self.sess.run(tf.global_variables_initializer())
            self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.message1 = tf.placeholder(tf.float32, [None, 512], name='message1')
        self.message1_ = tf.placeholder(tf.float32, [None, 512], name='message1_')
        self.message2 = tf.placeholder(tf.float32, [None, 512], name='message2')
        self.message2_ = tf.placeholder(tf.float32, [None, 512], name='message2_')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            s = tf.reshape(self.s, (-1, self.nplace, self.featureslen, 1))
            self.ec1 = self.my_leaky_relu(batch_norm(name='ebn1')(self.conv2d(s, 32, k_h=1, k_w=self.featureslen, name='e_conv1')))
            # self.ec2 = tf.nn.relu(batch_norm(name='ebn2')(self.conv2d(self.ec1, 64, k_h=1, k_w=1, name='e_conv2')))
            self.ec3 = self.my_leaky_relu(
                batch_norm(name='ebn3')(self.conv2d(self.ec1, 16, k_h=1, k_w=1, name='e_conv3')))
            e0 = tf.layers.flatten(self.ec3)
            e1 = tf.layers.dense(e0, 512, activation=self.lrelu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.e1 = tf.layers.dropout(e1, 0)
            e1 = tf.concat([self.e1, self.message1,self.message2], 1)
            e2 = tf.layers.dense(e1, 256, activation=self.lrelu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            e2 = tf.layers.dropout(e2, 0)
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            s_ = tf.reshape(self.s_, (-1, self.nplace, self.featureslen, 1))
            self.tc1 = self.my_leaky_relu(batch_norm(name='tbn1')(self.conv2d(s_, 32, k_h=1, k_w=self.featureslen, name='t_conv1')))
            # self.tc2 = tf.nn.relu(batch_norm(name='tbn2')(self.conv2d(self.tc1, 64, k_h=1, k_w=1, name='t_conv2')))
            self.tc3 = self.my_leaky_relu(
                batch_norm(name='tbn3')(self.conv2d(self.tc1, 16, k_h=1, k_w=1, name='t_conv3')))
            t0 = tf.layers.flatten(self.tc3)
            t1 = tf.layers.dense(t0, 512, activation=self.lrelu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.t1 = tf.layers.dropout(t1, 0)
            t1 = tf.concat([self.t1, self.message1_,self.message2_], 1)
            t2 = tf.layers.dense(t1, 256, activation=self.lrelu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            t2 = tf.layers.dropout(t2, 0)
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t3')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def my_leaky_relu(slef, x):
        return tf.nn.leaky_relu(x, alpha=0.05)

    def conv2d(self, input_, output_dim=8,
               k_h=1, k_w=11, d_h=1, d_w=1, stddev=0.02,
               name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def lrelu(self, x, leak=0.05, name="lrelu"):
        return tf.maximum(x, leak * x)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def get_message(self, observation,islearn=False,istarget=False):
        if islearn:
            if istarget:
                qval = self.sess.run(self.t1, feed_dict={self.s_: observation})
            else:
                qval = self.sess.run(self.e1, feed_dict={self.s: observation})
        else:
            observation = observation[np.newaxis, :]
            qval = self.sess.run(self.e1, feed_dict={self.s: observation})
        return qval

    def choose_action(self, observation,message1,message2):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            # np.reshape(self.sess.run([self.q_eval, self.e0], feed_dict={self.s: observation})[1], (2500, 8))
            actions_value, conv = self.sess.run([self.q_eval, self.ec3], feed_dict={self.s: observation,self.message1:message1,self.message2:message2,})
            action = np.argmax(actions_value)
        else:
            np.random.seed(int(time.time() * 1e7) % 10000000)
            action = np.random.randint(0, self.n_actions)
        return action

    def get_memory(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        return batch_memory

    def learn(self,batch_memory,m1,m1_,m2,m2_):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        _, cost, qt, q = self.sess.run(
            [self._train_op, self.loss, self.q_target, self.q_eval_wrt_a],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
                self.message1:m1,
                self.message1_:m1_,
                self.message2: m2,
                self.message2_: m2_,
            })
        # print(cost)

        self.cost_his.append(cost)

        # increasing epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his[1000:])), self.cost_his[1000:])
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

