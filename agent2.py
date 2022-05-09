import numpy as np
import tensorflow as tf
import time
from agentDQN import DeepQNetwork

class DQN2:
    def __init__(
            self,
            n_actions1,
            n_actions2,
            n_features,
            sess,
            epsilon=1.0,
    ):
        self.n_actions1 = n_actions1
        self.n_actions2 = n_actions2
        self.n_features = n_features
        self.RL1 = DeepQNetwork(n_actions1, n_features,
                          sess=sess,
                          name='dre',
                          learning_rate=0.01,
                          reward_decay=0.9,
                          replace_target_iter=200,
                          memory_size=5000,
                          )
        self.RL2 = DeepQNetwork(n_actions2, n_features,
                          sess=sess,
                          name='dis',
                          learning_rate=0.01,
                          reward_decay=0.9,
                          replace_target_iter=200,
                          memory_size=5000,
                          )
        self.RL3 = DeepQNetwork(3, n_features,
                                sess=sess,
                                name='choose',
                                learning_rate=0.01,
                                reward_decay=0.9,
                                replace_target_iter=200,
                                memory_size=5000,
                                )
        self.RL1.epsilon=epsilon
        self.RL2.epsilon=epsilon
        self.RL3.epsilon=epsilon

    def store_transition(self, s, a, r, s_):
        r1=r[0]+r[2]+r[4]+r[5]
        r2 = r[0] + r[2] + r[4]+r[1]+r[3]
        r3 = r[0] + r[2] + r[4]
        self.RL1.store_transition(s, a[0], r1, s_)
        self.RL2.store_transition(s, a[1], r2, s_)
        self.RL3.store_transition(s, a[2], r3, s_)

    def choose_action(self, observation):
        m1 = self.RL1.get_message(observation)
        m2 = self.RL2.get_message(observation)
        m3 = self.RL3.get_message(observation)
        action1=self.RL1.choose_action(observation, m2,m3)
        action2=self.RL2.choose_action(observation, m1,m3)
        action3 = self.RL3.choose_action(observation, m1, m2)
        return action1,action2,action3

    def learn(self):
        batch_memory1 = self.RL1.get_memory()
        batch_memory2 = self.RL2.get_memory()
        batch_memory3 = self.RL3.get_memory()
        m1 = self.RL1.get_message(batch_memory1[:, :self.n_features],islearn=True,istarget=False)
        m1_ = self.RL1.get_message(batch_memory1[:, -self.n_features:], islearn=True, istarget=True)
        m2 = self.RL2.get_message(batch_memory2[:, :self.n_features], islearn=True, istarget=False)
        m2_ = self.RL2.get_message(batch_memory2[:, -self.n_features:], islearn=True, istarget=True)
        m3 = self.RL3.get_message(batch_memory3[:, :self.n_features], islearn=True, istarget=False)
        m3_ = self.RL3.get_message(batch_memory3[:, -self.n_features:], islearn=True, istarget=True)
        self.RL1.learn(batch_memory1,m2,m2_,m3,m3_)
        self.RL2.learn(batch_memory2,m1,m1_,m3,m3_)
        self.RL3.learn(batch_memory3, m1, m1_,m2,m2_)
