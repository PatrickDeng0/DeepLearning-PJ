import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import features
import ob_util
from run import load_model, transform_pc


class SimpleStrategy:
    def __init__(self, bid_price, ask_price, pred):
        """
        :param bid_price: numpy.array of best bid
        :param ask_price: numpy.array of best ask
        :param pred: numpy.array of pred, whose elements is in {1,-1}
        """
        self.pred = pred
        self.ask_price = copy.copy(ask_price)
        self.bid_price = copy.copy(bid_price)
        self.wealth = np.zeros_like(self.ask_price)
        self.position = np.zeros_like(self.ask_price)
        self.cash0 = 100000
        self.cost_rate = 0.0001
        self.method = 1
        self.money = np.zeros_like(self.ask_price)
        self.ret_strat = np.zeros_like(self.ask_price)

        # preprocess for price==0
        ask_price[ask_price == 0] = np.nan
        bid_price[bid_price == 0] = np.nan
        refined_ask_price = pd.Series(ask_price).ffill()
        refined_bid_price = pd.Series(bid_price).ffill()
        self.mid_price = (0.5 * (refined_ask_price + refined_bid_price)).values

    def get_ret(self, cost_rate=0.0001, method=1, cash0=100000):
        """
        :param cost_rate: float
        :param method: 1==all in/all out; 2==1 share
        :param cash0: initial cash
        :return: ret_strat: numpy.array with same length as bid price
        """
        self.cash0 = cash0
        self.cost_rate = cost_rate
        self.method = method
        self.money += self.cash0

        # calculate the first buying time
        ini_buy = self.pred.argmax()
        self.ini_buy = ini_buy
        while self.ask_price[ini_buy] == 0:
            ini_buy += self.pred[ini_buy + 1:].argmax() + 1
            if ini_buy > len(self.ask_price):
                print("No action could be taken")
                return self.ret_strat

        for i in range(ini_buy, len(self.bid_price)):
            # buy
            if (self.pred[i] == 1) and (self.money[i - 1] >= 0) and (self.bid_price[i] > 0):
                if self.method == 1:
                    self.position[i] = self.position[i - 1] + self.money[i - 1] / (
                            self.bid_price[i] * (1 + self.cost_rate))  # buy all
                    self.money[i] = 0
                else:
                    self.position[i] = self.position[i - 1] + 1  # buy one
                    self.money[i] = self.money[i - 1] - self.bid_price[i] * (1 + self.cost_rate)
                self.wealth[i] = self.money[i] + self.position[i] * self.bid_price[i]

            # sell
            elif (self.pred[i] == -1) and (self.position[i - 1] >= 0) and (self.ask_price[i] > 0):
                if self.method == 1:
                    self.position[i] = 0
                    self.money[i] = self.money[i - 1] + self.position[i - 1] * self.ask_price[i] * (1 - self.cost_rate)
                else:
                    self.position[i] = self.position[i - 1] - 1
                    self.money[i] = self.money[i - 1] + self.ask_price[i] * (1 - self.cost_rate)
                self.wealth[i] = self.money[i] + self.position[i] * self.ask_price[i]


            # no action
            else:
                self.money[i] = self.money[i - 1]
                self.position[i] = self.position[i - 1]
                self.wealth[i] = self.money[i] + self.position[i] * self.mid_price[i]

            self.ret_strat[i] = self.wealth[i] / self.cash0 - 1

        return self.ret_strat[ini_buy:]

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
    ret_stra = SimpleStrategy(np.array(d['bid_px1']), np.array(d['ask_px1']), np.array(d['pred']))

    fig = plt.figure(figsize=(15, 10))
    plt.plot(ret_stra.get_ret(cost_rate=0)[:5000], label='No transaction cost')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.0001)[:5000], label='Transaction cost = 0.01%')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.0005)[:5000], label='Transaction cost = 0.05%')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.001)[:5000], label='Transaction cost = 0.1%')
    plt.grid()
    plt.legend()

    ret_stra.flush()

    fig = plt.figure(figsize=(15, 10))
    plt.plot(ret_stra.get_ret(cost_rate=0, method=2)[:5000], label='No transaction cost')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.0001, method=2)[:5000], label='Transaction cost = 0.01%')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.0005, method=2)[:5000], label='Transaction cost = 0.05%')
    ret_stra.flush()
    plt.plot(ret_stra.get_ret(cost_rate=.001, method=2)[:5000], label='Transaction cost = 0.1%')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lag = 50
    test_date = 20190604
    symbol = 'AMD'
    midwin = 15
    xwin = 50

    f_prefix = 'logs/{}/LSTMs_midwin{}_xwin{}_rate0.0001'.format(symbol, midwin, xwin)
    ob_file = './data/test_data/{}_ob_{}.csv'.format(symbol, test_date)
    trx_file = './data/test_data/{}_trx_{}.csv'.format(symbol, test_date)
    ob = pd.read_csv(ob_file)
    trx = pd.read_csv(trx_file)

    data = features.all_features(ob, trx, lag, include_ob=True)
    a, b, c = ob_util.generate_test_dataset(data, xwin, midwin)

    pca, ss, mod = load_model(f_prefix)
    test_x = transform_pc(a, pca, ss)
    pred = mod.predict(test_x).argmax(1)

    plot(pd.DataFrame({"bid_px1": b, "ask_px1": c, "pred": pred - 1}))
