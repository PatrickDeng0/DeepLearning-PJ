#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def pred_trans(pred_test):
    pred = np.zeros(len(pred_test))
    for i in range(len(pred_test)):
        if pred_test[i][0]>pred_test[i][1]:
            pred[i] = -1
        else:
            pred[i] = 1
    return pred

def time_merge(df, time_track, pred_test):
    pred = pred_trans(pred_test)
    time_track['pred'] = pred
    d = df.merge(time_track, on = 'time')
    return d

def trading_strategy(bid_price, ask_price, pred, cost_rate, method):
    #predict increasing, buy ask1
    #predict decreasing, sell bid1
    #not considering latency but considering transaction cost
    #bid_price is the best bid price
    #ask_price is the best ask price
    #pred is the prediction signal
    #cost rate is the transaction cost rate
    #method is the way of building position: 1==all in/all out; 2==1 share
    cash0 = 100000#the initial cash
    money = np.zeros(len(bid_price)) #it records the cash we hold in the process
    ret_stra = np.zeros(len(bid_price)) #it records the return of the strategy
    
    #begins from first buy signal
    ini_buy = list(pred).index(max(list(pred)))
    s0 = bid_price[ini_buy]#this is to find the first buy signal, and determine the first buy price 

    money[0:len(money)] = cash0
    wealth = np.zeros(len(bid_price))
    position = np.zeros(len(bid_price))
    for i in range(ini_buy, len(bid_price)):
        
        if method==1:
        
            if (pred[i]==1) and (money[i-1]>=0): #if buy signal appears, and there is enough money to buy
                position[i] = position[i-1]+money[i-1]*(1-cost_rate)/bid_price[i]#buy all
                money[i] = 0
                ret_stra[i] =((money[i]+position[i]*bid_price[i])-cash0)/cash0
                wealth[i] = money[i]+position[i]*bid_price[i]

            elif pred[i]==-1 and position[i-1]>=0 :#if sell signal appears and there is enough position to sell
                position[i] = 0
                money[i] = money[i-1]+position[i-1]*ask_price[i]*(1-cost_rate)
                ret_stra[i] = (money[i]+position[i]*ask_price[i]-cash0)/cash0
                wealth[i] = money[i]+position[i]*ask_price[i]
            else:
                money[i] = money[i-1]
                position[i] = position[i-1]
                ret_stra[i] = (money[i]+position[i]*(bid_price[i]/2+ask_price[i]/2)-cash0)/cash0
                wealth[i] = money[i]+position[i]*(bid_price[i]/2+ask_price[i]/2)
        
        else:

            if (pred[i]==1) and (money[i-1]>=0): #if buy signal appears, and there is enough money to buy
                position[i] = position[i-1]+1#buy one
                money[i] = money[i-1]-bid_price[i]*(1+cost_rate)
                ret_stra[i] =((money[i]+position[i]*bid_price[i])-cash0)
                wealth[i] = money[i]+position[i]*bid_price[i]

            elif pred[i]==-1 and position[i-1]>=0 :#if sell signal appears and there is enough position to sell
                position[i] = position[i-1]-1
                money[i] = money[i-1]+ask_price[i]*(1-cost_rate)
                ret_stra[i] = (money[i]+position[i]*ask_price[i]-cash0)
                wealth[i] = money[i]+position[i]*ask_price[i]
            else:
                money[i] = money[i-1]
                position[i] = position[i-1]
                ret_stra[i] = (money[i]+position[i]*(bid_price[i]/2+ask_price[i]/2)-cash0)
                wealth[i] = money[i]+position[i]*(bid_price[i]/2+ask_price[i]/2)
            
    return ret_stra

def plot(d):
    ret_stra1 = trading_strategy(np.array(d['bid_price_1'])/10000, np.array(d['ask_price_1'])/10000, np.array(d['pred']), 0, 1)
    ret_stra2 = trading_strategy(np.array(d['bid_price_1'])/10000, np.array(d['ask_price_1'])/10000, np.array(d['pred']), 0.0001, 1)
    ret_stra3 = trading_strategy(np.array(d['bid_price_1'])/10000, np.array(d['ask_price_1'])/10000, np.array(d['pred']), 0.0005, 1)
    ret_stra4 = trading_strategy(np.array(d['bid_price_1'])/10000, np.array(d['ask_price_1'])/10000, np.array(d['pred']), 0.001, 1)
 
    fig = plt.figure(figsize=(15,10))
    plt.plot(ret_stra1, label = 'No transaction cost')
    plt.plot(ret_stra2, label = 'Transaction cost = 0.01%')
    plt.plot(ret_stra3, label = 'Transaction cost = 0.05%')
    plt.plot(ret_stra4, label = 'Transaction cost = 0.1%')

    plt.legend()


# In[ ]:
import numpy as np
np.random.sample([])