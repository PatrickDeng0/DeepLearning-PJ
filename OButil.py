import pandas as pd
import numpy as np
import datetime as dt
import csv

# Global Variables
Q_TIME, Q_BID, Q_BIDSIZ, Q_ASK, Q_ASKSIZ = 0, 1, 2, 3, 4
T_TIME, T_SIZE, T_PRICE = 0, 1, 2


def t_s(time):
    t = time.split(":")
    return float(t[0]) * 3600 + float(t[1]) * 60 + float(t[2])


class OrderBook:
    def __init__(self, depth=5):
        self.depth = depth
        self.bids = {}
        self.asks = {}
        self.bid_prices = []
        self.ask_prices = []
        self.time = 0

    # Update best bid and ask price, for convenience in comparison
    # And Update the history of this order book, finally output to a csv
    def update(self):
        self.ask_prices = sorted(list(self.asks.keys()))
        self.bid_prices = sorted(list(self.bids.keys()), reverse=True)

    # Get the mid price of current order book
    def get_mid_price(self):
        if len(self.ask_prices) == 0 or len(self.bid_prices) == 0:
            return 0
        else:
            return (self.ask_prices[0] + self.bid_prices[0]) / 2

    # Update OB due to quote
    def handle_quote(self, quote):
        self.time = quote[Q_TIME]
        # Update bids
        self.bids[quote[Q_BID]] = quote[Q_BIDSIZ] * 100
        for price in self.bid_prices:
            if price > quote[Q_BID]:
                del self.bids[price]

        # Update asks
        self.asks[quote[Q_ASK]] = quote[Q_ASKSIZ] * 100
        for price in self.ask_prices:
            if price < quote[Q_ASK]:
                del self.asks[price]

        # Update best_price
        self.update()

    # For order book update when the trade is sell
    def sell_trade_update(self, trade_price, trade_size):
        # Sell limit order executed, now ask order book would change. Priority is descended by prices
        filled_size = 0
        for price in self.ask_prices:
            # If the price on ask order book is lower than the trade, then it must be eaten by the trade
            # So we accumulate the total numbers of orders eaten
            if price < trade_price:
                filled_size += self.asks[price]
                del self.asks[price]
            # Now if price is equal, we let the original amount of orders minus the accumulated orders
            elif price == trade_price:
                if filled_size < trade_size:
                    remain = self.asks[price] + filled_size - trade_size
                    if remain > 0:
                        self.asks[price] = remain
                    else:
                        del self.asks[price]
            else:
                break

    # For order book update when the trade is sell
    def buy_trade_update(self, trade_price, trade_size):
        # Buy limit order executed, now bid order book would change. Priority is increased by prices
        filled_size = 0
        for price in self.bid_prices:
            # If the price on ask order book is higher than the trade, then it must be eaten by the trade
            # So we accumulate the total numbers of orders eaten
            if price > trade_price:
                filled_size += self.bids[price]
                del self.bids[price]
            # Now if price is equal, we let the original amount of orders minus the accumulated orders
            elif price == trade_price:
                if filled_size < trade_size:
                    remain = self.bids[price] + filled_size - trade_size
                    if remain > 0:
                        self.bids[price] = remain
                    else:
                        del self.bids[price]
            else:
                break

    # Update OB due to trade
    def handle_trade(self, trade):
        self.time = trade[T_TIME]

        # Get the direction of this trade, and update the order book
        # direct = -1: "Sell" limit order, 1: "buy" limit order (According to Lobster)
        direct = None
        trade_price = trade[T_PRICE]
        trade_size = trade[T_SIZE]
        if len(self.ask_prices) > 0 and trade_price >= self.ask_prices[0]:
            direct = -1
            self.sell_trade_update(trade_price, trade_size)
        elif len(self.bid_prices) > 0 and trade_price <= self.bid_prices[0]:
            direct = 1
            self.buy_trade_update(trade_price, trade_size)
        else:
            pass
        # Update best_price and history
        self.update()
        return direct

    def show_order_book(self):
        def cut_depth(prices):
            while len(prices) < self.depth:
                prices.append(np.nan)
            return prices

        ask_prices = cut_depth(self.ask_prices.copy())
        bid_prices = cut_depth(self.bid_prices.copy())
        res = []
        for i in range(self.depth):
            res.extend([ask_prices[i], self.asks.get(ask_prices[i], np.nan), bid_prices[i],
                        self.bids.get(bid_prices[i], np.nan)])
        return np.array(res)


def preprocess_data(quote_dir, trade_dir, filename):
    current = dt.datetime.now()
    df_quote = pd.read_csv(quote_dir)
    df_trade = pd.read_csv(trade_dir)

    df_quote = df_quote[['TIME_M', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ']].values
    df_trade = df_trade[['TIME_M', 'SIZE', 'PRICE']].values

    vt_s = np.vectorize(t_s)
    df_quote[:, Q_TIME] = vt_s(df_quote[:, Q_TIME])
    df_trade[:, T_TIME] = vt_s(df_trade[:, T_TIME])

    def time_selection(data):
        end_time = t_s("16:00:00")
        time_line = data[:, 0]
        return data[time_line <= end_time]

    df_quote = time_selection(df_quote)
    df_trade = time_selection(df_trade)
    n_trade = len(df_trade)
    n_quote = len(df_quote)

    order_book = OrderBook(depth=5)

    def is_quote_next(trade_idx, quote_idx):
        if df_trade[trade_idx][0] > df_quote[quote_idx][0]:
            return True
        else:
            return False

    trade_index = 0
    quote_index = 0

    def handle_trade(trade_idx, rec):
        current_trade = df_trade[trade_idx]
        order_book.handle_trade(current_trade)
        if 34200 < current_trade[0] < 57600:
            rec.writerow(order_book.show_order_book())

    def handle_quote(quote_idx):
        order_book.handle_quote(df_quote[quote_idx])

    with open(filename, 'w', newline='') as file:
        recorder = csv.writer(file, delimiter=',')
        while trade_index < n_trade and quote_index < n_quote:
            if is_quote_next(trade_index, quote_index):
                handle_quote(quote_index)
                quote_index += 1
            else:
                handle_trade(trade_index, recorder)
                trade_index += 1
        while trade_index < n_trade:
            handle_trade(trade_index, recorder)
            trade_index += 1
        while quote_index < n_quote:
            handle_quote(quote_index)
            quote_index += 1

    print('Time lapse:', (dt.datetime.now() - current).total_seconds())

    return


# Question 1:
# What is training dataset?
# Now we divide the dataset into separate epochs where length of epochs is the window size +1
# Window size as data input, the last line as for judge the movement of mid price
def convert_to_dataset(filename, window_size):
    data = pd.read_csv(filename, header=None).values
    num_epochs = data.shape[0] // (window_size + 1)
    epochs_data = data[:num_epochs * (window_size + 1)]
    epochs_data = epochs_data.reshape(num_epochs, window_size + 1, epochs_data.shape[1])
    X = epochs_data[:, :-1, :]

    # Y: 0 for downwards, 1 for upwards
    mid_prices = np.mean(epochs_data[:, -2:, :3:2], axis=2)
    Y = np.diff(mid_prices, axis=1).squeeze()
    return X, Y


if __name__ == '__main__':
    # testing
    data_dir = '../Data/'
    preprocess_data(data_dir + 'INTC_quote_20120621.csv', data_dir + 'INTC_trade_20120621.csv',
                    data_dir + 'orderbook.csv')
    X, Y = convert_to_dataset(data_dir + 'orderbook.csv', window_size=10)
