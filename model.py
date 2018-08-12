import pandas as pd
from threading import Thread,Lock
import datetime
import random
from collections import defaultdict
from heapq import *
#from kivy.app import App
#import tbb
import time
from geopy.distance import great_circle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
lock=Lock()
def dijkstra(t,f):
    edges = [
        ("Vazhakulam", "Avoly", 4),
        ("Avoly", "Vazhakulam", 4),
        ("Vazhakulam", "Nadukara", 6),
        ("Vazhakulam", "Ayavana", 11),
        ("Nadukara", "Arakuzha", 5),
        ("Nadukara", "Vazhakulam", 6),
        ("Arakuzha", "Perumballoor", 4),
        ("Arakuzha", "Nadukara", 5),
        ("Ayavana", "AnchapettyJn", 5),
        ("Ayavana", "Varappetty", 12),
        ("Ayavana", "Vazhakulam", 11),
        ("Perumballoor", "Arakuzha", 4),
        ("Anchapetty", "Puthuppady", 7),
        ("Anchapetty", "Ayavana", 5),
        ("Anchapetty", "Anicadu", 13),
        ("Puthuppady", "Anchapetty", 7),
        ("Puthuppady", "Karukadam", 4),
        ("Puthuppady", "Chalikkadavu", 6),
        ("Varappetty", "Karukadam", 5),
        ("Varappetty", "Mathirappilly", 4),
        ("Varappetty", "Kothamangalam", 8),
        ("Varappetty", "Ayavana", 12),
        ("Karukadam", "Mathirappilly", 3),
        ("Karukadam", "Varappetty", 5),
        ("Karukadam", "Puthuppady", 4),
        ("Mathirappilly", "Kothamangalam", 7),
        ("Mathirappilly", "Karukadam", 3),
        ("Mathirappilly", "Varapetty", 4),
        ("Anicadu", "Anchapetty", 13),
        ("Anicadu", "Chalikkadavu", 7),
        ("Kothamangalam", "Mathirappilly", 7),
        ("Kothamangalam", "Varappetty", 8),
        ("Chalikkadavu", "Kanam", 2),
        ("Chalikkadavu", "Puthuppady", 6),
        ("Chalikkadavu", "Anicadu", 7),
        ("Kizhakkekara", "Kanam", 4),
        ("Kanam", "Chalikkadavu", 2),
        ("Kanam", "Kizhakkekara", 4)
    ]
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))

    q, seen, mins = [(0,f,())], set(), {f: 0}
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf")

def addtime(min,ctime):
    t = round(min, 2)
    t = t * 3600
    s = str(datetime.timedelta(seconds=t))
    timeList = []
    timeList.append(s)
    timeList.append(ctime)
    sum = datetime.timedelta()
    for i in timeList:
        (h, m, s) = i.split(':')
        d = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        sum += d
    return (str(sum))

def zone():
    t = time.strftime("%H:%M:%S")  # get system time
    hr = t[0] + t[1]
    hr = int(hr)
    mini = t[3] + t[4]
    mini = int(mini)
    if(mini==0 and hr==7):
        z=1
    elif(mini==0 and hr==10):
        z=2
    elif(mini==0 and hr==16):
        z=3
    elif(mini==0 and hr==19):
        z=4
    elif(mini==0 and hr==22):
        z=5
    else:
        if( hr>=7 and hr<10 and mini>0):
            z=2
        elif( hr>=10 and hr<16 and mini>0):
            z=3
        elif( hr>=16 and hr<19 and mini>0):
            z=4
        elif( hr>=19 and hr<22 and mini>0):
            z=5
        else:
            z=1
    return z

def lati_longi(loc):
    lock.acquire()
    lat_long = {'Kothamangalam': [10.060190, 76.635083],
                'Mathirappilly': [10.045843, 76.616721],
                'Karukadam': [10.033244, 76.609500],
                'Puthuppady': [10.011139, 76.603722],
                'Vazhakulam': [9.946958, 76.635900],
                'Avoly': [9.958699, 76.625166],
                'Anicadu': [9.968679, 76.607780],
                'Kizhakkekara': [9.977876, 76.592166],
                'Muvattupuza': [9.989423, 76.578975],
                'Arakuzha': [9.927757, 76.602278],
                'Perumballoor': [9.953572, 76.592763]
                }
    lock.release()
    return lat_long[loc]

def datasets1(current_loc,places,v,dataset,dest):
    path=dataset
    places1 = places
    data = pd.read_csv(path)
    model = RandomForestRegressor()
    l = ['Time_interval']+places1
    predictors =np.array(l)
    y = data.Average_speed
    X = data[predictors]
    model.fit(X, y)
    q = zone()
    d=[q]
    for j in v:
        d.append(j)
    l = lati_longi(current_loc)
    g = lati_longi('Muvattupuza')
    dist =great_circle(l,g).kilometers
    tym=(dist / model.predict([d]))
    calc_time(tym[0], q, current_loc, dest)
def wttime(tym,count):
    path="count.csv"
    data=pd.read_csv(path)
    model = RandomForestRegressor()
    predictors=(['Time_interval','Count'])
    y = data.Average_wt_time
    X = data[predictors]
    model.fit(X, y)
    d=[[tym,count]]
    return model.predict(d)
def calc_time(tym,q,current_loc,dest):
    arrival_time=addtime(tym,time.strftime("%H:%M:%S"))
    if q==1:
        count=random.randint(1,30)
    elif q==2:
        count = random.randint(90, 200)
    elif q==3:
        count = random.randint(20, 120)
    elif q==4:
        count = random.randint(100, 250)
    elif q==5:
        count=random.randint(20,100)
    wait_time=wttime(q,count)
    print("Assuming the number of vehicles count to be: ",count)
    print("The time you can cross the traffic is: ",addtime(wait_time[0]/3600 ,arrival_time))
    if wait_time >=100:
        ss=dijkstra(current_loc, dest)[1]
        ss=str(ss)
        m=['(',')',',',"'"]
        k=""
        for i in ss:
            if i not in m:k+=i
        k=k.split(" ")
        route="->".join(k)
        print("The most optimal path is ",route)
def model():
    places1=["Kothamangalam","Mathirappilly","Karukadam","Puthuppady"]
    places2=['Vazhakulam','Avoly','Anicadu','Kizhakkekara']
    places3=["Arakuzha","Perumballoor"]
    places=places2+places1+places3
    print(places1)
    print(places2)
    print(places3)
    current_loc=input("Enter the current location(any one from above list): ")
    dest=input("Enter the destination: ")
    v=[]
    if current_loc in places1 and dest not in places1:
        dataset="datasets1.csv"
        n=places1.index(current_loc)
        for i in range(n,len(places1)):
            print("Enter the speed at place ", places1[i])
            x = int(input())
            v.append(x)
            current_loc=places1[i]
            datasets1(current_loc, places1[n:i+1],v,dataset,dest)
    elif current_loc in places2 and dest not in places2:
        n = places2.index(current_loc)
        dataset="datasets2.csv"
        for i in range(n,len(places2)):
            print("Enter the speed at place:", places2[i])
            x = int(input())
            v.append(x)
            current_loc = places2[i]
            datasets1(current_loc, places2[n:i+1],v,dataset,dest)
    elif current_loc in places3 and dest not in places1:
        dataset="datasets3.csv"
        n = places3.index(current_loc)
        for i in range(n,len(places3)):
            print("Enter the speed at place ", places3[i])
            x = int(input())
            v.append(x)
            current_loc = places3[i]
            datasets1(current_loc, places3[n:i+1],v,dataset,dest)
    elif current_loc not in places:
        print("ERROR: Only places listed above is allowed")
        exit()
    else:
        print("No traffic block found in the route")
        print(current_loc,"->",dest)
        exit()
model()
