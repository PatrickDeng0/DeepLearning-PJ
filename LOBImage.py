# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:35:42 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
import OButil as ob

def generate_ob_image(ob_file_path, window_size=10, mid_price_window=5):
    '''
    :param ob_file_path: the order_book do not have column "mid", so we first will manually add the "mid" column
    :param window_size: same with ob.convert_to_dataset
    :param mid_price_window: same with ob.convert_to_dataset
    :return:
    '''
    data_df = pd.read_csv(ob_file_path)
    data_df["mid"] = (data_df["bid_px1"] + data_df["ask_px1"])/2
    X, y = ob.convert_to_dataset(data_df, window_size, mid_price_window)
    return ob.over_sample(X, y)

if __name__ == "__main__":
    ob_file_path = 'C:\\Users\\Administrator\\Desktop\\DeepLearningProject\\data\\order_book.csv'
    X, y = generate_ob_image(ob_file_path)