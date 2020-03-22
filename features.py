import pandas as pd
import numpy as np

"""
All feature functions assume the same number of rows in order book and trades
i.e. order book only updates upon new trades
Order book columns: ask_px1, ask_sz1, bid_px1, bid_sz1, ask_px2, ask_sz2, bid_px2, bid_sz2...
Transactions columns: price, size, direction (-1 for sell, 1 for buy, na for mid price transactions)
"""

# Column indices
TRANS_PX, TRANS_SZ, TRANS_DIR = 0, 1, 2


def order_flow(order_book_df, transaction_df, lag=50):
    """
    order flow = the ratio of the volume of market buy(sell) orders arriving in the prior n observations
                 to the resting volume of ask(bid) limit orders at the top of book
    This feature is constructed according to the paper.
    Intuition: an increase in this ratio will more likely deplete the best ask level and the mid-price will up-tick,
               and vice-versa for a down-tick.
    actual spread = ask_price_1 - bid_price_1
    relative spread =  (actual spread / mid price) * 10000
    actual market imbalance = the volume of market buy orders arriving in the prior n observations
                            - the volume of market sell orders arriving in the prior n observations
    This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P8
    relative market imbalance = actual market imbalance / actual spread
    This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P8
    Intuition: a small actual spread combined with a strongly positive actual market imbalance
               would indicate buying pressure.
    """
    flow = pd.concat([order_book_df[['ask_sz1', 'bid_sz1']], transaction_df[['tx_size', 'tx_direction']]], axis=1)
    flow['buy_vol'] = 0
    flow['sell_vol'] = 0
    flow.loc[flow['tx_direction'] == 1, 'buy_vol'] = flow['tx_size']
    flow.loc[flow['tx_direction'] == -1, 'sell_vol'] = flow['tx_size']
    flow['order_flow_buy'] = flow['buy_vol'].rolling(lag).sum() / flow['ask_sz1']
    flow['order_flow_sell'] = flow['sell_vol'].rolling(lag).sum() / flow['bid_sz1']

    mid_price = (order_book_df['ask_px1'] + order_book_df['bid_px1']) / 2
    flow['actual_spread'] = order_book_df['ask_px1'] - order_book_df['bid_px1']
    flow['relative_spread'] = (flow['actual_spread'] / mid_price) * 1000

    flow['actual_mkt_imb'] = flow['buy_vol'].rolling(lag).sum() - flow['sell_vol'].rolling(lag).sum()
    flow['relative_mkt_imb'] = flow['actual_spread'] / flow['actual_mkt_imb']

    return flow.drop(columns=['ask_sz1', 'bid_sz1', 'tx_size', 'tx_direction', 'buy_vol', 'sell_vol'])


def liquidity_imbalance(order_book_df, depth):
    """
    liquidity imbalance at level i = ask_vol_i / (ask_vol_i + bid_vol_i)
    This feature is constructed according to the ppt.
    """
    liq_imb = {}
    for i in range(depth):
        a, b = order_book_df['ask_sz{}'.format(i + 1)], order_book_df['bid_sz{}'.format(i + 1)]
        liq_imb['liq_imb_{}'.format(i + 1)] = a / (a + b)

    return pd.DataFrame(liq_imb)


def relative_mid_trend(order_book_df):
    """
    First, construct a variation on mid-price where the average of the bid and ask prices is weighted
    according to their inverse volume. Then, divide this variation by common mid price.
    This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P10
    Intuition: a larger relative_mid_price_trend would more likely lead to a up-tick.
    """

    nom = (order_book_df['ask_px1'] / order_book_df['ask_sz1'] + order_book_df['bid_px1'] / order_book_df['bid_sz1'])
    den = (1 / order_book_df['ask_sz1'] + 1 / order_book_df['bid_sz1'])
    mid_price_inv_vol_weighted = nom / den
    mid_price = (order_book_df['ask_px1'] + order_book_df['bid_px1']) / 2

    return mid_price_inv_vol_weighted / mid_price


def volatility(order_book_df, lag=50):
    """
    The volatility is the standard deviation of the last n mid prices returns then divided by 100
    This feature is derived from paper: Angelo Ranaldo..._Order aggressiveness in limit order book markets...P4
    """
    mid_price = (order_book_df['ask_px1'] + order_book_df['bid_px1']) / 2
    mid_price_return = mid_price.shift(-1) - mid_price
    volatility_look_ahead = (mid_price_return.rolling(lag).std()) / 100
    return volatility_look_ahead.shift(1)

#
# def aggressiveness(order_book_df, transaction_df, lag=50):
#     """
#     bid(ask) limit order aggressiveness = the ratio of bid(ask) limit orders submitted at no lower(higher) than
#                                                        the best bid(ask) prices in the prior n observations
#                                                     to total bid(ask) limit orders submitted in prior 50 observations
#     This feature is derived from book: Irene Aldridge_High-frequency trading...(2013) P186
#     Intuition: The higher the ratio, the more aggressive is the trader in his bid(ask) to capture the best
#                available price and the more likely the trader is to believe that the price is about to
#                move away from the mid price.
#     """
#     df = pd.concat([order_book_df[['ask_sz1', 'bid_sz1']], transaction_df[['tx_size', 'tx_direction']]], axis=1)
#
#
#     is_aggr_sell = (df['tx_direction'] == -1 & df['tx_price'] <= df['ask_px1'].shift(1))
#
#     df['is_aggr_sell'] = is_aggr_sell
#     df['aggr_sell_size'] = 0
#     df.loc[df['is_aggr_sell'], 'aggr_sell_size'] = df['tx_size']
#
#     df['sell_tx_size'] = 0
#     df.loc[df['tx_direction'] == -1, 'sell_tx_size'] = df['tx_size']
#
#     agr_sell_ratios = df['aggr_sell_size'].rolling(50).sum() / df['sell_tx_size'].rolling(50).sum()
#
#     # bid limit order aggressiveness
#     if_bid_sbmt_agr_mid1 = (df['type_direction'] == 1)
#     if_bid_sbmt_agr_mid2 = (df['price'] >= df['bid_price_1'].shift(1))
#     if_bid_sbmt_agr = (if_bid_sbmt_agr_mid1 & if_bid_sbmt_agr_mid2)
#
#     df['if_bid_sbmt_agr'] = if_bid_sbmt_agr
#     if_bid_sbmt_agr_index = df[df['if_bid_sbmt_agr'] == True].index
#     df['bid_vol_sbmt_agr'] = 0
#     df.loc[if_bid_sbmt_agr_index, 'bid_vol_sbmt_agr'] = df['size']
#
#     if_bid_sbmt_index = df[df['type_direction'] == 1].index
#     df['bid_vol_sbmt'] = 0
#     df.loc[if_bid_sbmt_index, 'bid_vol_sbmt'] = df['size']
#
#     df['lo_agr_bid'] = df['bid_vol_sbmt_agr'].rolling(50, min_periods=1).sum() / df[
#         'bid_vol_sbmt'].rolling(50, min_periods=1).sum()
#
#     # df['lo_agr_bid'].fillna(method='ffill', inplace=True)
#     all_features['lo_agr_bid'] = df['lo_agr_bid']
#
#
# def features_new(orderbook_file, message_file):
#     messages = pd.read_csv(message_file, header=None)
#     limit_order_book = pd.read_csv(orderbook_file, header=None)
#
#     messages.columns = ['time', 'type', 'order ID', 'size', 'price', 'direction']
#     limit_order_book.columns = ['{}_{}_{}'.format(s, t, l) for l in range(1, 6) for s in ['ask', 'bid'] for t in
#                                 ['price', 'vol']]
#
#     all_features = pd.concat([messages.time, limit_order_book], axis=1)
#     total_data = pd.concat([messages, limit_order_book], axis=1)
#     total_data['mid_price'] = (total_data.loc[:, 'ask_price_1'] + total_data.loc[:, 'bid_price_1']) / 2
#
#     price_movement = np.sign(total_data['mid_price'].shift(-1) - total_data['mid_price'])
#     total_data['mid_price_mov'] = np.zeros(len(total_data))
#     total_data['mid_price_mov'][0:len(total_data) - 1] = price_movement[0:len(total_data) - 1]  # the last one is nan
#     total_data['mid_price_mov'][len(total_data) - 1] = 0
#     total_data['label'] = total_data['mid_price_mov']
#
#     # -- new feature 10: effective spread
#     '''
#     The effective spread is computed as difference between the latest trade price and midprice
#                                         divided by midprice, then times 1000.
#     This feature is derived from book: Irene Aldridge_High-frequency trading...(2013) P191
#     Intuition: The effective spread measures how far, in percentage terms, the latest realized price
#                fell away from the simple mid price.
#     '''
#     if_lastest_trade_index = total_data[total_data['type'] == 4].index
#     if_not_lastest_trade_index = total_data[total_data['type'] != 4].index
#     total_data['lastest_trade_price'] = 0
#     total_data.loc[if_lastest_trade_index, 'lastest_trade_price'] = total_data['price']
#     total_data.loc[if_not_lastest_trade_index, 'lastest_trade_price'] = np.nan
#     total_data['lastest_trade_price'].fillna(method='ffill', inplace=True)
#
#     total_data['effective_spread'] = (total_data['lastest_trade_price'] / total_data['mid_price'] - 1) * 1000
#
#     # total_data['effective_spread'].fillna(method='ffill', inplace=True)
#     all_features['effective_spread'] = total_data['effective_spread']
#
#     # -- new feature 11: ILLIQ
#     """
#     The illiquidity is computed as the ratio of absolute stock return to its dollar volume.
#
#     This feature is derived from Amihud (2002)
#
#     """
#
#     total_data['mid_price_ret'] = np.log(total_data['mid_price']) - np.log(total_data['mid_price'].shift(1))
#     total_data['ret_over_volume'] = abs(total_data['mid_price_ret']) / (
#             total_data['ask_vol_1'] + total_data['bid_vol_1'])
#     total_data['ILLIQ'] = total_data['ret_over_volume'].rolling(50, min_periods=1).sum()
#
#     all_features['ILLIQ'] = total_data['ILLIQ']
#
#     # -- new feature 12: relative volume
#     """
#     Relative volume is computed as the ratio of current volume to the historical average volume
#     """
#
#     for i in range(1, 6):
#         total_data['rel_ask_vol_' + str(i)] = total_data['ask_vol_' + str(i)] / total_data['ask_vol_' + str(i)].rolling(
#             50, min_periods=1).mean()
#         total_data['rel_bid_vol_' + str(i)] = total_data['bid_vol_' + str(i)] / total_data['bid_vol_' + str(i)].rolling(
#             50, min_periods=1).mean()
#
#         all_features['rel_bid_vol_' + str(i)] = total_data['rel_bid_vol_' + str(i)]
#         all_features['rel_ask_vol_' + str(i)] = total_data['rel_ask_vol_' + str(i)]
#
#     # -- new feature 13: volume depth
#
#     """
#     Volume depth is computed as the ratio of best volume to the sum of all depth volume
#     """
#     total_data['depth_ask_vol'] = total_data['ask_vol_1'] / (
#             total_data['ask_vol_1'] + total_data['ask_vol_2'] + total_data['ask_vol_3'] + total_data['ask_vol_4'] +
#             total_data['ask_vol_5'])
#     total_data['depth_bid_vol'] = total_data['bid_vol_1'] / (
#             total_data['bid_vol_1'] + total_data['bid_vol_2'] + total_data['bid_vol_3'] + total_data['bid_vol_4'] +
#             total_data['bid_vol_5'])
#
#     all_features['depth_ask_vol'] = total_data['depth_ask_vol']
#     all_features['depth_bid_vol'] = total_data['depth_bid_vol']
#
#     # -- new feature 14: volume rank
#     """
#     volume rank is computed as the rank of current volume with respect to the previous 50days volume
#     """
#
#     rollrank = lambda x: (x.argsort().argsort()[-1] + 1.0) / len(x)
#
#     for i in range(1, 6):
#         total_data['rank_ask_vol_' + str(i)] = total_data['ask_vol_' + str(i)].rolling(50, min_periods=1).apply(
#             rollrank)
#         total_data['rank_bid_vol_' + str(i)] = total_data['bid_vol_' + str(i)].rolling(50, min_periods=1).apply(
#             rollrank)
#
#         total_data['rank_ask_vol_' + str(i)] = total_data['rank_ask_vol_' + str(i)].fillna(method='ffill', axis=0)
#         total_data['rank_bid_vol_' + str(i)] = total_data['rank_bid_vol_' + str(i)].fillna(method='ffill', axis=0)
#         total_data['rank_ask_vol_' + str(i)] = np.clip(total_data['rank_ask_vol_' + str(i)], 0, 1)
#         total_data['rank_bid_vol_' + str(i)] = np.clip(total_data['rank_bid_vol_' + str(i)], 0, 1)
#
#         all_features['rank_bid_vol_' + str(i)] = total_data['rank_bid_vol_' + str(i)]
#         all_features['rank_ask_vol_' + str(i)] = total_data['rank_ask_vol_' + str(i)]
#
#     # -- new feature 15: ask bid volume correlation
#     """
#     ask bid volume correlation is comupted as 50 days time series correlation between ask and bid volume for each level
#     """
#
#     for i in range(1, 6):
#         total_data['corr_vol_' + str(i)] = total_data['ask_vol_' + str(i)].rolling(50, min_periods=1).corr(
#             total_data['bid_vol_' + str(i)])
#
#         total_data['corr_vol_' + str(i)] = total_data['corr_vol_' + str(i)].fillna(method='ffill', axis=0)
#         total_data['corr_vol_' + str(i)] = np.clip(total_data['corr_vol_' + str(i)], -1, 1)
#
#         all_features['corr_vol_' + str(i)] = total_data['corr_vol_' + str(i)]
#
#     ##ADD technical indicators
#     total_data['ma7'] = total_data['mid_price'].rolling(7, min_periods=1).mean()
#     total_data['ma21'] = total_data['mid_price'].rolling(21, min_periods=1).mean()
#     all_features['ma7'] = total_data['ma7']
#     all_features['ma21'] = total_data['ma21']
#
#     # Create DMA
#     total_data['DMA'] = total_data['mid_price'].rolling(10, min_periods=1).mean() - total_data['mid_price'].rolling(50,
#                                                                                                                     min_periods=1).mean()
#     total_data['AMA'] = total_data['DMA'].rolling(10, min_periods=1).mean()
#     all_features['DMA'] = total_data['DMA']
#     all_features['AMA'] = total_data['AMA']
#
#     """
#            #Create DPO
#     offset = int(20/2+1)
#     total_data['DPO'] = total_data['mid_price'][offset:]-total_data['mid_price'].rolling(20,min_periods = 1).mean()[:len(total_data)-offset]
#     all_features['DPO'] = total_data['DPO']
#     """
#
#     # Create EMA
#     K = 2 / (20 + 1)
#     total_data['EMA'] = np.zeros(len(total_data))
#     for i in range(len(total_data)):
#         if i == 0:
#             total_data['EMA'][i] = total_data['mid_price'][i]
#         else:
#             total_data['EMA'][i] = total_data['mid_price'][i] * K + total_data['EMA'][i - 1] * (1 - K)
#     all_features['EMA'] = total_data['EMA']
#
#     """
#             #Create ENV
#     Length=14
#     Width=0.06
#     ENV=np.zeros(len(total_data))
#     MiddleLine=total_data['mid_price'].rolling(Length, min_periods = 1).mean()
#     UpperLine=MiddleLine*(1+Width)
#     LowerLine=MiddleLine*(1-Width)
#
#     for i in range(len(total_data)):
#         if total_data['mid_price'][i]>UpperLine[i]:
#             ENV[i]=1
#         elif total_data['mid_price'][i]<LowerLine[i]:
#             ENV[i]=-1
#     total_data['ENV'] = ENV
#     all_features['ENV'] = total_data['ENV']
#     """
#
#     # Create PSY
#     length = 12
#     PSYValue = np.full(len(total_data), 50.0)  # the PSY value for first 12 day is set to be 50
#     for i in range(length, len(total_data)):
#         PSYValue[i] = sum(np.array(total_data['mid_price'][i - length + 1:i + 1]) > np.array(
#             total_data['mid_price'][i - length:i])) / length * 100
#     total_data['PSY'] = PSYValue
#     all_features['PSY'] = total_data['PSY']
#
#     # Create ROC
#     ROCValue = np.zeros(len(total_data['mid_price']))  # the PVI value for 12 day is set to be 0
#     ROCValue[length:len(total_data['mid_price'])] = (np.array(
#         total_data['mid_price'][length:len(total_data['mid_price'])]) - np.array(
#         total_data['mid_price'][0:len(total_data['mid_price']) - length])) / np.array(
#         total_data['mid_price'][0:len(total_data['mid_price']) - length]) * 100
#     total_data['ROC'] = ROCValue
#     all_features['ROC'] = total_data['ROC']
#
#     # Create RSI
#     length = 14
#     RSIValue = np.zeros(len(total_data['mid_price']))
#     DiffofPrice = np.zeros(len(total_data['mid_price']))
#     DiffofPrice[1:len(total_data['mid_price'])] = np.array(
#         total_data['mid_price'][1:len(total_data['mid_price'])]) - np.array(
#         total_data['mid_price'][0:len(total_data['mid_price']) - 1])
#     RSIValue[0:length] = 50  # the RSI value for first 14 days is set to be 50
#     for i in range(length, len(total_data)):
#         temp = DiffofPrice[i - length:i]
#         RSIValue[i] = sum(temp[temp > 0]) / sum(abs(temp)) * 100  # one way of calculation
#     total_data['RSI'] = RSIValue
#     all_features['RSI'] = total_data['RSI']
#
#     # Create BIAS
#     Length = 6
#     BIASValue = np.zeros(len(total_data))
#     BIASValue = (total_data['mid_price'] - total_data['mid_price'].rolling(Length, min_periods=1).mean()) / total_data[
#         'mid_price'].rolling(Length, min_periods=1).mean() * 100
#     total_data['BIAS'] = BIASValue
#     all_features['BIAS'] = total_data['BIAS']
#
#     # Create CMO
#     Length = 14
#     Price = total_data['mid_price']
#     len1 = len(Price)
#     CMOValue = np.zeros(len1)
#     DiffofPrice = np.zeros(len1)
#     DiffofPrice[1:] = np.diff(Price)
#     for i in range(Length, len1):
#         Temp = DiffofPrice[i - Length + 1:i]
#         CMOValue[i] = (np.sum(Temp[Temp > 0]) - np.sum(np.abs(Temp[Temp < 0]))) / np.sum(np.abs(Temp)) * 100
#     total_data['CMO'] = CMOValue
#     all_features['CMO'] = total_data['CMO']
#
#     # Create BOLL
#     Length = 20
#     Width = 2
#     Price = total_data['mid_price']
#     len1 = len(total_data)
#     MiddleLine = np.zeros(len1)
#     UpperLine = np.zeros(len1)
#     LowerLine = np.zeros(len1)
#
#     MiddleLine = total_data['mid_price'].rolling(Length, min_periods=1).mean()
#     UpperLine[:Length - 1] = MiddleLine[:Length - 1]
#     LowerLine[:Length - 1] = MiddleLine[:Length - 1]
#     for i in range(Length - 1, len1):
#         UpperLine[i] = MiddleLine[i] + Width * np.std(Price[i - Length + 1:i])
#         LowerLine[i] = MiddleLine[i] - Width * np.std(Price[i - Length + 1:i])
#
#     total_data['Upper_line'] = UpperLine
#     total_data['Lower_line'] = LowerLine
#     all_features['Upper_line'] = total_data['Upper_line']
#     all_features['Lower_line'] = total_data['Lower_line']
#
#     """
#             # Create MACD
#     Length=26
#     Width=0.06
#     ENV=np.zeros(len(total_data));
#     MiddleLine=total_data['mid_price'].rolling(Length, min_periods = 1).mean()
#     UpperLine=MiddleLine*(1+Width)
#     LowerLine=MiddleLine*(1-Width)
#
#     for i in range(len(total_data)):
#         if total_data['mid_price'][i]>UpperLine[i]:
#             ENV[i]=1
#         elif total_data['mid_price'][i]<LowerLine[i]:
#             ENV[i]=-1
#     total_data['26ema'] = ENV
#
#     Length=12
#     Width=0.06
#     ENV=np.zeros(len(total_data));
#     MiddleLine=total_data['mid_price'].rolling(Length, min_periods = 1).mean()
#     UpperLine=MiddleLine*(1+Width)
#     LowerLine=MiddleLine*(1-Width)
#     for i in range(len(total_data)):
#         if total_data['mid_price'][i]>UpperLine[i]:
#             ENV[i]=1
#         elif total_data['mid_price'][i]<LowerLine[i]:
#             ENV[i]=-1
#     total_data['12ema'] = ENV
#             #total_data['26ema'] = pd.ewm(total_data['mid_price'], span=26)
#             #total_data['12ema'] = pd.ewm(total_data['mid_price'], span=12)
#             #total_data['26ema'] = pd.DataFrame(total_data['mid_price']).ewm(span=26)
#             #total_data['12ema'] = pd.DataFrame(total_data['mid_price']).ewm(span=12)
#     total_data['MACD'] = (total_data['12ema']-total_data['26ema'])
#
#             # Create Bollinger Bands
#             #total_data['20sd'] = pd.stats.moments.rolling_std(total_data['mid_price'],20)
#             #total_data['upper_band'] = total_data['ma21'] + (total_data['20sd']*2)
#             #total_data['lower_band'] = total_data['ma21'] - (total_data['20sd']*2)
#
#             # Create Momentum
#
#
#
#     all_features['26ema'] = total_data['26ema']
#     all_features['12ema'] = total_data['12ema']
#     all_features['MACD'] = total_data['MACD']
#     """
#     # self._all_features['20sd'] = total_data['20sd']
#     # self._all_features['upper_band'] = total_data['upper_band']
#     # self._all_features['lower_band'] = total_data['lower_band']
#     total_data['momentum'] = total_data['mid_price'] - 1
#     all_features['momentum'] = total_data['momentum']
#     ###ADD fourier transform
#     close_fft = np.fft.fft(np.asarray(total_data['mid_price'].tolist()))
#     fft_df = pd.DataFrame({'fft': close_fft})
#     all_features['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
#     all_features['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
#
#     # --
#     all_features['label'] = total_data['label']
#     all_features = all_features.dropna()
#
#     all_features['label'] = all_features['label'].astype(int)
#     return all_features
