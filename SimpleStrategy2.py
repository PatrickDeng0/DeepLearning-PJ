import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Notes: in this version, we did not check the accuracy of
 - 'pred_test'
 - 'time_merge'
'''

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


class SimpleStrategy:
    def __init__(self, bid_price, ask_price, pred):
        '''
        :param bid_price: numpy.array of best bid
        :param ask_price: numpy.array of best ask
        :param pred: numpy.array of pred, whose elements is in {1,-1}
        '''
        self.pred = pred
        self.ask_price = ask_price
        self.bid_price = bid_price
        self.wealth = np.zeros_like(self.ask_price)
        self.position = np.zeros_like(self.ask_price)
        self.cash0 = 100000
        self.cost_rate = 0.0001
        self.method = 1
        self.money = np.zeros_like(self.ask_price)
        self.ret_strat = np.zeros_like(self.ask_price)

    def get_ret(self, cost_rate=0.0001, method=1, cash0=100000):
        '''
        :param cost_rate: float
        :param method: 1==all in/all out; 2==1 share
        :param cash0: initial cash
        :return: ret_strat: numpy.array with same length as bid price
        '''
        self.cash0 = cash0
        self.cost_rate = cost_rate
        self.method = method
        self.money += self.cash0

        # calculate the first buying time
        ini_buy = self.pred.argmax()
        while(self.ask_price[ini_buy]==0):
            ini_buy += self.pred[ini_buy+1:].argmax() + 1
            if ini_buy>len(self.ask_price):
                print("No action could be taken")
                return self.ret_strat


        for i in range(ini_buy, len(self.bid_price)):
            # buy
            if (self.pred[i] == 1) and (self.money[i - 1] >= 0) and (self.ask_price[i] > 0):
                if self.method == 1:
                    self.position[i] = self.position[i - 1] + self.money[i - 1] / (
                                self.ask_price[i] * (1 + self.cost_rate))  # buy all
                    self.money[i] = 0
                else:
                    self.position[i] = self.position[i - 1] + 1  # buy one
                    self.money[i] = self.money[i - 1] - self.ask_price[i] * (1 + self.cost_rate)

            # sell
            elif (self.pred[i] == -1) and (self.position[i - 1] >= 0) and (self.bid_price[i] > 0):
                if self.method == 1:
                    self.position[i] = 0
                    self.money[i] = self.money[i - 1] + self.position[i - 1] * self.bid_price[i] * (1 - self.cost_rate)
                else:
                    self.position[i] = self.position[i - 1] - 1
                    self.money[i] = self.money[i - 1] + self.bid_price[i] * (1 - self.cost_rate)

            # no action
            else:
                self.money[i] = self.money[i - 1]
                self.position[i] = self.position[i - 1]

            # calculate the net value at each time spot
            if self.position[i] >= 0:
                self.wealth[i] = self.money[i] + self.position[i] * self.ask_price[i]
            else:
                self.wealth[i] = self.money[i] + self.position[i] * self.bid_price[i]

            self.ret_strat[i] = self.wealth[i] / self.cash0 - 1

        return self.ret_strat

    def get_position(self):
        return self.position

    def get_wealth(self):
        return self.wealth

    def get_money(self):
        return self.money

    def flush(self):
        self.wealth *= 0
        self.money *= 0
        self.position *= 0
        self.ret_strat *= 0



def plot(d):
    ret_stra = SimpleStrategy(np.array(d['bid_px1']), np.array(d['ask_px1']),
                                 np.array(d['pred']))

    fig = plt.figure(figsize=(15, 10))
    plt.plot(ret_stra.get_ret(cost_rate=0), label='No transaction cost')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.0001), label='Transaction cost = 0.01%')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.0005), label='Transaction cost = 0.05%')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.001), label='Transaction cost = 0.1%')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    '''
    tested with data in the Data\orderbook.csv
    '''
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\DeepLearningProject\Data\orderbook.csv")
    d = pd.DataFrame({"bid_px1":df["bid_px1"], "ask_px1":df["ask_px1"], "pred":np.random.choice([-1, 1], len(df["bid_px1"]))})

    plot(d)