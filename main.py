from scipy.optimize import minimize
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt

start = datetime.datetime(2021,1,1)
end = datetime.datetime(2021,1,21)

ceva = yf.download('CEVA',start,end)
google = yf.download('GOOGL',start,end)
tesla = yf.download('TSLA',start,end)
zom = yf.download('ZOM',start,end)

stocks = pd.concat([ceva['Close'],google['Close'],tesla['Close'],zom['Close']],axis=1)
stocks.columns = ['CEVA','GOOGLE','TESLA','ZOMEDICA']

returns = stocks/stocks.shift(1)
logReturns = np.log(returns)

np.random.seed(1)

noOfPortfolios = 10000
weight = np.zeros((noOfPortfolios,4))
expectedReturns = np.zeros(noOfPortfolios)
expectedVolality = np.zeros(noOfPortfolios)
sharpeRatio = np.zeros(noOfPortfolios)

meanLogRet = logReturns.mean()
Sigma = logReturns.cov()

for k in range(noOfPortfolios):
    w = np.array(np.random.random(4))
    w = w/np.sum(w)
    weight[k,:]=w
    expectedReturns[k]=np.sum(meanLogRet*w)
    expectedVolality[k]=np.sqrt(np.dot(w.T,np.dot(Sigma,w)))
    sharpeRatio[k]=expectedReturns[k]/expectedVolality[k]

maxIndex = sharpeRatio.argmax()
print(weight[maxIndex,:])
print(expectedReturns,expectedVolality)
plt.figure(figsize=(12,12))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Log Returns')
plt.scatter(expectedVolality,expectedReturns,c=sharpeRatio)
plt.scatter(expectedVolality[maxIndex],expectedReturns[maxIndex],c='red')
plt.colorbar(label='SharpeRatio')
#plt.show()  


#markowitz
def negSR(w):
    w= np.array(w)
    R=np.sum(meanLogRet*w)
    V = np.sqrt(np.dot(w.T,np.dot(Sigma,w)))
    SR = R/V
    return -1*SR
def checkSumToOne(w):
    return np.sum(w)-1
w0 = [.25,.25,.25,.25]
bounds = ((0,1),(0,1),(0,1),(0,1))
constraints = ({'type':'eq','fun':checkSumToOne})
w_opt = minimize(negSR,w0,method='SLSQP',bounds=bounds,constraints=constraints)
print(w_opt)

#frontier
returns = np.linspace(0,.07,50)
volatlity_opt = []

def minimizeMyVolatility(w):
    w= np.array(w)
    V = np.sqrt(np.dot(w.T,np.dot(Sigma,w)))
    return V
def getReturn(w):
    w = np.array(w)
    R = np.sum(meanLogRet*w)
    return R


for r in returns:
    constraints = ({'type':'eq','fun':checkSumToOne},
               {'type':'eq','fun':lambda w:getReturn(w) - r})
    opt = minimize(minimizeMyVolatility,w0,method='SLSQP',bounds=bounds,constraints=constraints)
    volatlity_opt.append(opt['fun'])
plt.plot(volatlity_opt,returns)
plt.show()