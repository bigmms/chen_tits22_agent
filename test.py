import matplotlib.pyplot as plt
from env import Env
from agentDQNsingle import DeepQNetwork
import os
import agent
import agentdouble
import datetime
import tensorflow as tf
import argparse
import numpy as np
import xlrd
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
env = Env(day=3, placenum=2000)


def get_keywordtime():
    wb = xlrd.open_workbook(filename="keywords.xlsx")
    sheet = wb.sheet_by_index(0)
    data = [[sheet.row(i)[j].value for j in range(sheet.ncols)] for i in range(sheet.nrows)]
    d = dict()
    k = []
    for x in data:
        d[x[1]] = x[2]
        k.append(x)
    return k, d
def test(end, nowtime, daytime, ini,Budgetlevel,mytype,RL):
    key,dic=get_keywordtime()
    lat,lng=ini
    elat, elng=end
    t0 = datetime.datetime.now()
    for dayt in range(7):
        daytime=dayt
        for Bud in range(5):
            # Bud=3
            Budgetlevel=Bud
            observation = env.reset(elat=elat,
                                    elng=elng,
                                    nowtime=nowtime,
                                    daytime=daytime,
                                    lat=lat,
                                    lng=lng,
                                    Budgetlevel=Budgetlevel,
                                    mytype=mytype,
                                    )
            score = 0
            ac = []
            rlist = []
            state = []
            print(env.userinfo1)
            n = env.tot
            while True:
                action = RL.choose_action(observation)
                if type(action)==np.int64:
                    if action >= 0 and action <= 15:
                        action3 = 0
                        tmp = action
                    elif action >= 16 and action <= 31:
                        action3 = 1
                        tmp = action - 16
                    elif action >= 32 and action <= 47:
                        action3 = 2
                        tmp = action - 32
                    action1 = int(tmp // env.actionsize1)
                    action2 = tmp % env.actionsize2
                elif len(action) == 2:
                    action1 = action[0]
                    action2 = action[1]
                    action3 = -1
                elif len(action) == 3:
                    action1 = action[0]
                    action2 = action[1]
                    action3 = action[2]
                observation_, reward, rl, done = env.step(observation, action1, action2,action3)
                score = score + reward
                observation = observation_
                state.append(observation)
                ac.append((action1, action2,action3))
                rlist.append(rl)
                if done:
                    score = score / n
                    t1 = datetime.datetime.now()
                    print("  score:", score, "  len:", len(env.record), "  time:", t1 - t0)
                    f1=0
                    f2=0
                    f3=0
                    for rec in env.record:
                        recc= rec[0]
                        if int(recc[1])==0:
                            f1+=abs(int(recc[4])-Bud)
                        if rec[1]<=-50:
                            f3+=1
                        if int(recc[2])==1:
                            f2+=1
                    print("B-:", f1, "  T+:", f2, "  GG:", f3)
                    if f1<=3 and f2>=2 and f3==0 and score>30 and len(env.record)==nowtime:
                        print("dayt%d_Bud%dXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"%(dayt,Bud))
                        result = env.result[:env.placenum]
                        for rec in env.record:
                            if rec[0][1] == 0.0:
                                ar = 'Dining'
                            elif rec[0][1] == 1.0:
                                ar = 'Touring'
                            elif rec[0][1] == 2.0:
                                ar = 'Accommodation'
                            print(rec[0][0], '(%s)'%(ar),"  Type:", key[int(result[rec[-1]][2])][1],
                                  "  Pricelevel:", rec[0][4],"\nCoordinate:%f,%f"%(rec[0][6],rec[0][7]))
                        print(rlist)
                        print(ac)
                        x = [i[0][6] for i in env.record]
                        y = [i[0][7] for i in env.record]
                        x.insert(0, env.userinfo1[3])
                        y.insert(0, env.userinfo1[4])
                        x.insert(len(x), env.userinfo1[7])
                        y.insert(len(y), env.userinfo1[8])
                        fig = plt.figure()
                        ax1 = fig.add_subplot(111)
                        ax1.plot(y, x, c='r', marker='o')
                        plt.savefig('save_graph/ours_dayt%d_Bud%d.png'%(dayt,Bud))
                        plt.clf()
                        aaa=1
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--place', type=str, default='oldLA')
    parser.add_argument('--endloc', type=float, default=(34.182957, -118.315241))
    parser.add_argument('--iniloc', type=float, default=(34.017027, -118.113996))
    parser.add_argument('--tottime', type=float, default=7)  # 678
    parser.add_argument('--daytime', type=float, default=0)
    parser.add_argument('--Budgetlevel', type=float, default=2)
    parser.add_argument('--mytype', type=float, default=[10,18])
    opt = parser.parse_args()
    with tf.Session() as sess:
        RL = agent2.DQN2(env.actionsize1, env.actionsize2, env.statesize, sess=sess,epsilon=0.0)
        saver = tf.train.Saver()
        placename=opt.place
        ss=opt.place
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir/%s' % (ss)))
        env.loaddata(keywordversion=2,loadname=placename)
        test(end=opt.endloc,
             nowtime=opt.tottime,
             daytime=opt.daytime,
             ini=opt.iniloc,
             Budgetlevel=opt.Budgetlevel,
             mytype=opt.mytype,
             RL=RL)
    step = 1

