from geopy.geocoders import Nominatim
import random
import pickle
import geopy.distance
import numpy as np
import xlrd
# from sklearn.cluster import KMeans
import datetime
import matplotlib.pyplot as plt
import math
import argparse

class Datapro():
    def __init__(self, placenum=500):
        self.actionsize1 = 4
        self.actionsize2 = 4
        self.result = None
        self.placenum = placenum
        self.dis = [2.0, 5.0, 10.0, 25.0]
        self.dre = [i for i in range(0, 360, int(360 / self.actionsize1))]

    def getdata(self, placelist, keywordversion=1):
        self.keywordversion = keywordversion
        self.keyword, self.keywordtime = self.get_keywordtime()
        self.result = []
        for placess in placelist:
            result = []
            for p in placess:
                search_result = []
                with open('SearchResult/%s1v%d.pickle' % (p, self.keywordversion), 'rb') as file:
                    search_result.append(pickle.load(file))
                with open('SearchResult/%s2v%d.pickle' % (p, self.keywordversion), 'rb') as file:
                    search_result.append(pickle.load(file))
                with open('SearchResult/%s3v%d.pickle' % (p, self.keywordversion), 'rb') as file:
                    search_result.append(pickle.load(file))
                cnt = 0
                for i, res in enumerate(search_result):
                    for key, value in res.items():
                        for loc in value:
                            if not loc.get('rating'): continue
                            tmp = []
                            lat = loc['geometry']['location']['lat']
                            lng = loc['geometry']['location']['lng']
                            dis = 0
                            # dis = geopy.distance.distance(self.inilocation, (lat, lng)).kilometers
                            tmp.append(loc['name'])
                            tmp.append(i)
                            tmp.append(cnt)
                            tmp.append(loc['rating'] * loc['rating'] * np.sqrt(loc['user_ratings_total']))
                            if loc.get('price_level'):
                                tmp.append(loc['price_level'])
                            else:
                                tmp.append(0)
                            # tmp.append(self.keywordtime[key])  # placetime
                            tmp.append(1)
                            tmp.append(lat)
                            tmp.append(lng)
                            tmp.append(0)
                            tmp.append(dis)
                            tmp.append(0)
                            result.append(tmp)
                        cnt = cnt + 1

            result.sort()
            de = []
            for i in range(len(result) - 1):
                f = 0
                for j in range(len(result[i])):
                    if j == 0 or j == 6 or j == 7:
                        if result[i][j] != result[i + 1][j]:
                            f = 1
                            break
                if f == 0: de.append(i)
            result = np.array(result)
            result = np.delete(result, de, axis=0)
            result = list(result)
            result.sort(key=self.sortbyrating, reverse=True)
            de = []
            for i in range(len(result) - 1):
                f = 0
                for j in range(len(result[i])):
                    if j == 0 or j == 6 or j == 7:
                        if result[i][j] != result[i + 1][j]:
                            f = 1
                            break
                if f == 0: de.append(i)
            result = np.array(result)
            result = np.delete(result, de, axis=0)
            result = list(result)
            result.sort(key=self.sortbyrating, reverse=True)
            self.result.append(result)
        # [sum(self.placeinfo[]==i)for i in range(41)]
        # self.placeinfo = result[np.random.choice(int(len(result)*0.4), self.placenum, replace=False)]
        print("load data succeed")
        self.cluster()
        data = dict()
        data['dremat'] = self.dremat
        data['dismat'] = self.dismat
        data['placeinfo'] = self.result
        data['isolate'] = self.isolate
        data['nearby'] = self.nearby
        file = open('./dataset/LA%dv2.pickle' % (self.placenum), 'wb')
        pickle.dump(data, file)
        file.close()

    def sortbyrating(self, x):
        return float(x[3])

    def cluster(self):
        self.dismat = np.zeros((len(self.result), self.placenum, self.placenum), dtype=np.float32)
        self.dremat = np.zeros((len(self.result), self.placenum, self.placenum), dtype=np.float32)
        self.isolate = []
        self.nearby = []
        for i in range(len(self.result)):
            result = self.result[i][:self.placenum]
            # kk=[ k for k in result if k[1]=='2']
            x = np.array(result)[:,7]
            y = np.array(result)[:,6]
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            plt.xlabel('X')
            plt.ylabel('Y')
            ax1.scatter(x.astype(np.float32), y.astype(np.float32), c='r', marker='o')
            plt.show()
            isola = []
            t0 = datetime.datetime.now()
            nearby = []
            for x in range(len(result)):
                nb = [[[] for ii in range(4)] for jj in range(4)]
                for ddre in range(4):
                    for ddis in range(4):
                        ns = self.get_distance_point(result[x][6], result[x][7], self.dis[ddis], self.dre[ddre])
                        for id, ppp in enumerate(result):
                            tmp = geopy.distance.distance((ppp[6], ppp[7]), (ns.latitude, ns.longitude)).kilometers
                            if tmp < 2.0:
                                nb[ddre][ddis].append(id)
                for y in range(len(result)):
                    dis = geopy.distance.distance((result[x][6], result[x][7]), (result[y][6], result[y][7])).kilometers
                    dre = self.azimuthAngle(result[x][6], result[x][7], result[y][6], result[y][7])
                    self.dismat[i, x, y] = dis
                    self.dremat[i, x, y] = dre
                if sum(self.dismat[i, x] < 2) == 1:
                    isola.append(x)
                nearby.append(nb)
                t1 = datetime.datetime.now()
                print(x, t1 - t0, nb)
            # dis = np.array(dis).reshape(-1, 1)
            # estimator = KMeans(n_clusters=self.actionsize)
            # estimator.fit(dis)
            # centroids = estimator.cluster_centers_
            # self.dis = list(centroids[:, 0])
            # self.dis.sort()
            self.isolate.append(isola)
            self.nearby.append(nearby)
        print("preprocessing succeed")

    def get_distance_point(self, lat, lon, distance, direction):
        start = geopy.Point(lat, lon)
        d = geopy.distance.VincentyDistance(kilometers=distance)
        return d.destination(point=start, bearing=direction)

    def get_keywordtime(self):
        wb = xlrd.open_workbook(filename="keywords v%d.xlsx" % (self.keywordversion))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--place', type=str, default='TW')
    opt = parser.parse_args()
    datapro = Datapro(placenum=2000)
    datapro.getdata(placelist=[[opt.place]], keywordversion=2)
    a = 1
