import matplotlib.pyplot as plt
from env import Env
import pylab
from agentDQN import DeepQNetwork
import os
import datetime
import tensorflow as tf
import numpy as np
import argparse
from agent import DQN2
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
env = Env(day=3, placenum=2000)


def plot_cost(x):
    plt.clf()
    plt.plot(np.arange(len(x)), x)
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    city='LA'
    parser.add_argument('--place', type=str, default=city)
    opt = parser.parse_args()
    with tf.Session() as sess:
        RL = DQN2(env.actionsize1, env.actionsize2, env.statesize,sess=sess)
        saver = tf.train.Saver()
        env.loaddata(keywordversion=2, loadname=opt.place)
        step = 1
        scores = []
        episodes = []
        time = []
        rr = []
        badresult = []
        s1s,s2s,s3s=[],[],[]
        for episode in range(2000):
            state = []
            ac = []
            rlist = []
            # initial observation
            t0 = datetime.datetime.now()
            placeid = 0
            observation = env.reset(0, 0, 0, 0, lat=33.941530, lng=-118.408558, placeid=placeid, randomUser=True)
            print(env.userinfo1)
            n = env.tot
            score = 0
            s1,s2,s3=0,0,0
            if episode % 200 == 199:
                saver.save(sess, './checkpoint_dir/%s/MyModel'%(city))
            while True:
                # fresh env

                # RL choose action based on observation
                action = RL.choose_action(observation)
                action1 = action[0]
                action2 = action[1]
                action3 = action[2]

                # RL take action and get next observation and reward
                observation_, reward, rl, done = env.step(observation, action1, action2,action3)
                state.append(observation)
                ac.append(action)
                rlist.append(rl)
                score = score + reward
                r1 = rl[0] + rl[2] + rl[4] + rl[5]
                r2 = rl[0] + rl[2] + rl[4] + rl[1] + rl[3]
                r3 = rl[0] + rl[2] + rl[4]
                s1=s1+r1
                s2=s2+r2
                s3 = s3 + r3
                RL.store_transition(observation, action, rl, observation_)

                if step > 1000 and step % 2 == 0:
                    RL.learn()
                if episode == 1900:
                    a = 1
                observation = observation_

                # break while loop when end of this episode
                if done:
                    rlist = np.average(np.array(rlist), axis=0)
                    rr.append(rlist)
                    score = score / n
                    s1=s1/n
                    s2=s2/n
                    s3=s3/n
                    s1s.append(s1)
                    s2s.append(s2)
                    s3s.append(s3)
                    if score < -50:
                        badresult.append([episode, score, env.record, env.userinfo1, state, ac])
                    t1 = datetime.datetime.now()
                    time.append(t1 - t0)
                    scores.append(score)
                    episodes.append(episode)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./save_graph/reward-epicode_dqn.png")
                    print("episode:", episode, "  score:", score, "  len:", len(env.record), "  step:",
                          step, "  epsilon:", RL.RL1.epsilon, "  time:", time[-1])
                    print(env.record)
                    step += 1
                    break
                step += 1
        aaaaa=1
