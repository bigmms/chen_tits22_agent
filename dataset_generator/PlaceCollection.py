import geopy.distance
import numpy as np
import time
import requests
import pickle
import random
import xlrd
import argparse
key = [
    'Your Google map API key'
]

def get_distance_point(lat, lon, distance, direction):
    start = geopy.Point(lat, lon)
    d = geopy.distance.VincentyDistance(kilometers=distance)
    return d.destination(point=start, bearing=direction)

def coordgen(inilocation, radius, region):
    c = []
    s = get_distance_point(inilocation[0], inilocation[1], region, 270)
    s = get_distance_point(s.latitude, s.longitude, region, 0)
    for i in range(int(region // radius) + 2):
        for j in range(int(region // radius) + 2):
            ns = get_distance_point(s.latitude, s.longitude, radius * i * 2, 90)
            ns = get_distance_point(ns.latitude, ns.longitude, radius * j * 2, 180)
            if geopy.distance.distance(inilocation, (ns.latitude, ns.longitude)).kilometers <= region: c.append(
                (ns.latitude, ns.longitude))
            ns = get_distance_point(ns.latitude, ns.longitude, radius, 90)
            ns = get_distance_point(ns.latitude, ns.longitude, radius, 180)
            if geopy.distance.distance(inilocation, (ns.latitude, ns.longitude)).kilometers <= region: c.append(
                (ns.latitude, ns.longitude))
    return c

def get_keywordtime():
    wb = xlrd.open_workbook(filename="keywords v2.xlsx")
    sheet = wb.sheet_by_index(0)
    data = [[sheet.row(i)[j].value for j in range(sheet.ncols)] for i in range(sheet.nrows)]
    d = dict()
    k = []
    for x in data:
        d[x[1]] = x[2]
        k.append(x)
    keys = [[], [], []]
    for x in k:
        keys[int(x[0] - 1)].append(x)
    return keys, d

def get_location(inilocation=(34.040648,-118.246824), radius=4000, region=25000,name='LA'):
    coords = coordgen(inilocation, radius / 1000, region / 1000)
    key_words, _ = get_keywordtime()
    for i in range(3):
        result = dict()
        key_word = key_words[i]
        for kk in key_word:
            kw = kk[1]
            result[kw] = []
            for cr in coords:
                search_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=' + \
                             str(cr[0]) + ',' + str(cr[1]) + '&radius=' + str(radius) + '&type=' + kw \
                             + '&keyword=' + kw + '&key=' + key[random.randint(0, len(key) - 1)]
                search_req = requests.get(search_url)
                search_res = search_req.json()
                result[kw].extend(search_res['results'])
                tmp = np.random.randint(60, high=80, size=None, dtype='l')
                print(kw, tmp)
                time.sleep(tmp)
                while search_res.get('next_page_token'):
                    search_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?pagetoken=' + search_res.get(
                        'next_page_token') + '&key=' + key[random.randint(0, len(key) - 1)]
                    search_req = requests.get(search_url)
                    search_res = search_req.json()
                    result[kw].extend(search_res['results'])
                    tmp = np.random.randint(50, high=60, size=None, dtype='l')
                    print(kw, tmp)
                    time.sleep(tmp)
                print(result[kw])
            file = open('./SearchResult/%s%dv2-radius%d-region%d.pickle' % (name,i + 1, radius, region), 'wb')
            pickle.dump(result, file)
            file.close()
    result = []
    return result
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--placelat', type=float, default=34.040648)
    parser.add_argument('--placelng', type=float, default=-118.246824)
    parser.add_argument('--radius', type=int, default=4000)
    parser.add_argument('--region', type=int, default=25000)
    parser.add_argument('--placename', type=str, default='LA')
    opt = parser.parse_args()
    get_location(inilocation=(opt.placelat,opt.placelng),radius=opt.radius,region=opt.region,name=opt.placename)
