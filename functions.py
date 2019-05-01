import numpy as np
import math
from sklearn import preprocessing


def Normalized(X):
    return preprocessing.normalize(X)


# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
    ans = []
    if x< -50:
        ans = 0
    elif x > 100:
        ans = 1
    else:
        ans= 1 / (1 + math.exp(-x))
    return ans


# returns an an n-day state representation ending at time t

def getState(closeprice, volume, t, n):
    d = t - n + 1
    block = closeprice[d:t + 1] if d >= 0 else -d * [closeprice[0]] + closeprice[0:t + 1] # pad with t0
    Volume = volume[d:t + 1] if d >= 0 else -d * [volume[0]] + volume[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
        #res.append(sigmoid(math.log(block[0 + 1],block[0])))
        #res.append(math.log(block[0 + 1],block[0])*100)
    for j in range(n - 1):
        #res.append((sigmoid((Volume[0 + 1] - Volume[0]))))
        #res.append(sigmoid(math.log(Volume[0 + 1],Volume[0])))
        res.append(math.log(Volume[0 + 1],Volume[0])*100)
    return np.array([res])


def getStockVolume(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[6]))

	return vec

def getStatetest(closeprice, volume, t, n):
    d = t - n + 1
    block = closeprice[d:t + 1] if d >= 0 else -d * [closeprice[0]] + closeprice[0:t + 1] # pad with t0
    Volume = volume[d:t + 1] if d >= 0 else -d * [volume[0]] + volume[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
        print(block[i + 1] - block[i])
    #for j in range(n - 1):
        #res.append((sigmoid((Volume[0 + 1] - Volume[0])/Volume[0])))
        #res.append(sigmoid(math.log(Volume[0 + 1],Volume[0])))
    return np.array([res])