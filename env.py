import pickle
import geopy.distance
import numpy as np
import xlrd
import datetime
import matplotlib.pyplot as plt
import math
from functools import cmp_to_key
def cmp1(i, j):
    if i[1] <= -50: return -1
    if j[1] <= -50: return 1
    x=i[0]
    y=j[0]
    if x[0] == y[0]:
        return (x[2]+x[4]) - (y[2]+y[4])
    else:
        return x[0] - y[0]
def cmp2(i, j):
    if i[1] <= -50: return -1
    if j[1] <= -50: return 1
    x=i[0]
    y=j[0]
    if x[2] == y[2]:
        return (x[0]+x[4]) - (y[0]+y[4])
    else:
        return x[2] - y[2]
def cmp3(i, j):
    if i[1] <= -50: return -1
    if j[1] <= -50: return 1
    x=i[0]
    y=j[0]
    if x[4] == y[4]:
        return (x[2]+x[0]) - (y[2]+y[0])
    else:
        return x[4] - y[4]
def cmp0(i, j):
    if i[1] <= -50: return -1
    if j[1] <= -50: return 1
    x=i[0]
    y=j[0]
    return sum(x) - sum(y)
class Env():
    def __init__(self, day=2, placenum=500):
        self.actionsize2 = 4
        self.actionsize1 = 4
        self.placeinfo = None
        self.result = None
        self.nowtime = 0
        self.placelistid = 0
        self.trip = [0, 1, 0, 1, 0, 1, 2]
        self.daytrip = 7
        self.daytime = 0
        self.day = day
        self.placenum = placenum
        self.resetuser()
        self.userinfo = [np.random.randint(0, self.actionsize1), self.day, self.Budgetlevel, 0, 0, self.nowtime,
                         self.daytime]
        self.statesize = self.placenum * (7 + len(self.userinfo))
        self.dis = [2.0, 5.0, 10.0, 25.0]
        self.dre = [i for i in range(0, 360, int(360 / self.actionsize1))]

    def resetuser(self, elat=0, elng=0, nowtime=0, daytime=0, lat=25.067618, lng=121.552652, age=20, season=2,
                  Budgetlevel=1, randomUser=False,mytype=None):
        if randomUser:
            self.inid = np.random.randint(0, self.placenum)
            while self.inid in self.isolate[self.placelistid]:
                self.inid = np.random.randint(0, self.placenum)
            x = self.placeinfo[self.inid]
            self.inilocation = (float(x[6]), float(x[7]))
            self.endid = np.random.randint(0, self.placenum)
            x = self.placeinfo[self.endid]
            self.endlocation = (float(x[6]), float(x[7]))
            self.Budgetlevel = np.random.randint(0, 4)
            n = np.random.randint(1, 6)
            self.mytype = np.random.choice([i for i in range(50)], n)
            self.daytime = np.random.randint(0, self.daytrip)
            self.nowtime = np.random.randint(self.daytrip, 36)

        else:
            self.inilocation = (lat, lng)
            if mytype==None:mytype=[]
            self.mytype=mytype
            self.endlocation = (elat, elng)
            self.Budgetlevel = Budgetlevel
            self.daytime = daytime
            self.nowtime = nowtime

    def loaddata(self, keywordversion=2, loadname=None):
        self.keywordversion = keywordversion
        self.keyword, self.keywordtime = self.get_keywordtime()
        self.result = []
        with open('dataset/%s2000v%d.pickle' % (loadname, self.keywordversion), 'rb') as file:
            data = pickle.load(file)
        self.dismat = data['dismat']
        self.dremat = data['dremat']
        self.isolate = data['isolate']
        self.result = data['placeinfo'][0]

        self.nearby = data['nearby']
        print("load data succeed")
        return

    def sortbyrating(self, x):
        return float(x[3])

    def reset(self, elat, elng, nowtime, daytime,mytype=None, lat=25.067618, lng=121.552652, age=20, season=2, Budgetlevel=1,
              placeid=0, randomUser=False):
        self.inid = None
        self.placelistid = placeid
        result = self.result[:self.placenum]
        self.placeinfo = np.array(result)
        self.resetuser(elat, elng, nowtime, daytime, lat, lng, age,
                       season, Budgetlevel, randomUser,mytype=mytype)
        self.placeXtrafficinfo = []
        self.tot = self.nowtime
        self.record = []
        self.placeinfo[:, 10] = 0
        self.placeinfo[:, 2] = np.array([int(int(self.placeinfo[i, 2]) in self.mytype) for i in range(self.placenum)])
        self.lat, self.lng = self.inilocation[0], self.inilocation[1]
        self.get_userinfo(np.random.randint(0, self.actionsize1))
        self.eatday = 0
        self.id = self.inid
        for x in self.placeinfo:
            x[8] = self.azimuthAngle(self.inilocation[0], self.inilocation[1], x[6], x[7])
            x[9] = geopy.distance.distance((x[6], x[7]), (self.inilocation[0], self.inilocation[1])).kilometers
            tmp = list(x)
            self.placeXtrafficinfo.append(tmp)
        state0 = np.array(self.placeXtrafficinfo)[:, 1:].astype(np.float32)
        state0 = self.makestate(state0)
        return state0

    def get_distance_point(self, lat, lon, distance, direction):
        start = geopy.Point(lat, lon)
        d = geopy.distance.VincentyDistance(kilometers=distance)
        return d.destination(point=start, bearing=direction)

    def getplace(self, s, a1, a2,a3):
        place = None
        rdre = 1
        rdis = a2
        max = float('-inf')
        id = 0
        tmp=[]
        dre = self.dre[a1]
        dis = self.dis[a2]
        ns = self.get_distance_point(self.lat, self.lng, dis, dre)
        rlist = [0, 0, 0, 0,0]
        if self.daytime >= self.daytrip:
            self.daytime = self.daytime - self.daytrip
        if len(self.record):
            if abs(self.dre[self.record[-1][2]] - dre) == 180: rdre = -1
        if len(self.record):
            for i in self.nearby[self.placelistid][self.id][a1][a2]:
                now = list(self.placeinfo[i])
                now = [float(p) for p in now[1:]]
                now.insert(0, self.placeinfo[i][0])
                r, rl = self.get_reward(s.reshape(self.placenum, -1)[i], now, i)
                tmp.append((rl,r,i,now))
        else:
            for i, x in enumerate(self.placeinfo):
                dis = geopy.distance.distance((x[6], x[7]), (ns.latitude, ns.longitude)).kilometers
                if dis < 2:
                    now = list(self.placeinfo[i])
                    now = [float(p) for p in now[1:]]
                    now.insert(0, self.placeinfo[i][0])
                    r, rl = self.get_reward(s.reshape(self.placenum, -1)[i], now, i)
                    tmp.append((rl, r, i, now))
        if len(tmp):
            if a3==0:tmp.sort(key=cmp_to_key(lambda x, y: cmp1(x, y)))
            elif a3==1:tmp.sort(key=cmp_to_key(lambda x, y: cmp2(x, y)))
            elif a3==2:tmp.sort(key=cmp_to_key(lambda x, y: cmp3(x, y)))
            elif a3==-1:tmp.sort(key=cmp_to_key(lambda x, y: cmp0(x, y)))
            rlist = tmp[-1][0]
            max = tmp[-1][1]
            id = tmp[-1][2]
            place = tmp[-1][3]
        return id, place, max, rlist, rdre * 30.0, rdis * self.tot * 0.0

    def step(self, s, a1, a2,a3):
        id, now1, r, rlist, rdre, rdis = self.getplace(s, a1, a2,a3)
        if not now1:
            r = -100
            if len(self.record) == 0:
                id = self.inid
            else:
                id = self.record[-1][-1]
        self.id = id
        now = list(self.placeinfo[id])
        now = [float(p) for p in now[1:]]
        now.insert(0, self.placeinfo[id][0])
        self.nowtime = self.nowtime - 1
        self.daytime = self.daytime + 1
        done = False
        if self.nowtime == 0:
            done = True
        if r <= -50:
            rlist = [r / 6, r / 6, r / 6, r / 6, r / 6,r / 6]
        else:
            rlist.append(rdre)
            rlist = [rlist[i] + 40 / 6. for i in range(6)]
            r += 40
            r = r + rdre + rdis
        if now1:
            self.record.append((now, r, a1, a2,a3, id))
        if id:
            self.lat, self.lng = now[6], now[7]
            self.get_userinfo(a1)
            self.placeinfo[id, 10] = 1 + int(self.placeinfo[id, 10])
            self.placeinfo[:, 8] = self.dremat[self.placelistid, id]
            self.placeinfo[:, 9] = self.dismat[self.placelistid, id]
        s_ = np.array(self.placeinfo)[:, 1:].astype(np.float32)
        s_ = self.makestate(s_)
        return s_, r, rlist[:6], done

    def get_reward(self, s, now, id):
        r1 = s[2]
        r2 = 3 - now[9]
        r3 = 1 - (abs(self.Budgetlevel - now[4]))
        r4 = 0
        dft = [x[-1] for x in self.record]
        r5 = sum(self.dismat[self.placelistid, id, dft]) - 10 * len(self.record)
        if self.nowtime == 1:
            dis = geopy.distance.distance((now[6], now[7]), (self.endlocation[0], self.endlocation[1])).kilometers
            r4 = 18 - dis
        f5 = 0
        r6 = 1 if now[2] == 1 else -0.1
        for x in self.record:
            if x[0][0] == now[0]:
                f5 = 1
        if f5: return -50, [0, 0, 0, 0,0]
        if self.trip[self.daytime] != now[1]: return -50, [0, 0, 0, 0,0]
        rlist=[r1 * 5.0, r2 * 1.0 + r4 * 5.0, r3 * 10, r5 * 0.1,r6*20]
        return sum(rlist), rlist

    def makestate(self, s):
        u = np.array(self.userinfo)
        st = s[:, [2, 7, 8]]
        std = np.std(st, axis=0, ddof=1)
        std[std == 0] = 1
        avg = np.average(st, axis=0)
        st = (st - avg) / std
        u[1] = u[1] / self.daytrip
        u[2] = (u[2] - avg[0]) / std[0]
        u[3] = (u[3] - avg[1]) / std[1]
        u[4] = (u[4] - avg[2]) / std[2]
        u[5] = u[5] / self.daytrip
        u[6] = u[6] / self.daytrip
        s[:, [2, 7, 8]] = st
        s = np.concatenate((s, np.tile(u[np.newaxis, :], (self.placenum, 1))), axis=1)
        s = np.delete(s, [4, 5, 6], axis=1)
        s = s.flatten()
        return s

    def get_userinfo(self, a1):
        dis = geopy.distance.distance((self.lat, self.lng), (self.endlocation[0], self.endlocation[1])).kilometers
        dre = self.azimuthAngle(self.lat, self.lng, self.endlocation[0], self.endlocation[1])
        self.userinfo = [a1, self.tot, self.Budgetlevel, dre, dis, self.nowtime, self.daytime]
        self.userinfo1 = [a1, self.day, self.Budgetlevel, self.inilocation[0], self.inilocation[1],
                          self.nowtime, self.daytime, self.endlocation[0], self.endlocation[1], dre, dis]

    def get_keywordtime(self):
        wb = xlrd.open_workbook(filename="keywords.xlsx")
        sheet = wb.sheet_by_index(0)
        data = [[sheet.row(i)[j].value for j in range(sheet.ncols)] for i in range(sheet.nrows)]
        d = dict()
        k = []
        for x in data:
            d[x[1]] = x[2]
            k.append(x)
        return k, d

    def azimuthAngle(self, y1, x1, y2, x2):
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        angle = 0.0
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1:
                angle = 0.0
            elif y2 < y1:
                angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        return (angle * 180 / math.pi)
