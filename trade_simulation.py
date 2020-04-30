import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import features
import ob_util as ob


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


def strategy_performance(model, order_book_df, transaction_df, window_size=10, mid_price_window=5, lag=50):
    X = features.all_features(order_book_df, transaction_df, lag, include_ob=True)
    test_df = order_book_df[lag - 1:]

    test_X, action_time = ob.generate_test_dataset(X, window_size, mid_price_window)
    test_X[:, :, -20:] = ob.normalize_ob(test_X[:, :, -20:])
    test_X = np.nan_to_num(test_X)
    pred = np.ones(len(test_df))
    pred[action_time] = model.predict(test_X).argmax(1)
    d = pd.DataFrame(
        {"bid_px1": test_df["bid_px1"], "ask_px1": test_df["ask_px1"], "pred": pred - 1}
    )
    return d
