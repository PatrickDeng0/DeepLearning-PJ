# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:32:44 2020

@author: Administrator
"""
import pandas as pd
import numpy as np
class IOBImage:
    def __init__(self, order_book):
        # preprocess: handling nan value
        # in the bid side, the reason for a current level is 0 is that the previous level is nan or zero, this has been proved. So the preprocess will turn all the nan value in bid to be zero
        # on the ask side, if current level is nan, then all the levels later will be nan too, so we could also replace nan with 0
        order_book = order_book.fillna(0)
        order_book["time"] = order_book.index
        self.order_book = order_book
    
    def _generate_ob_image(self, window_size=10):
        '''
        input: 
            - order_book: pandas.dataframe with a column "time"
            - window_size: int, backward periods(include the current time), for example, for window_size=10, at time = t, the corresponding time forming the picture is t, t-1, ...., t-9
        output: pandas.dataframe
                images, times
                <numpy.array,  shape=(window_size*20)>, <int>
        '''
        times = []
        images = []
        for i in range(window_size-1, len(self.order_book)):
            images.append(self.order_book.iloc[i-window_size+1:i+1, :-1].values)
            times.append(self.order_book["time"][i])
        self.images = np.stack(images)
        images_id = range(len(images))
        self.images_id_df = pd.DataFrame({"images_id":images_id, "times":times})
        
    def _generate_image_label(self, window_size=10):
        # use the Equation(4) in the paper 
        price = pd.DataFrame({"price":(self.order_book["bid_px1"]+self.order_book["ask_px1"])/2}).rolling(window_size).mean()
        returns = np.concatenate(price.iloc[2*window_size-1:, ].values/price.iloc[window_size-1:len(self.order_book)-window_size, ].values-1)
        times = range(window_size-1, len(self.order_book)-window_size)
        self.labels_df = pd.DataFrame({"returns":returns, "times":times})
    
    def get_X_y(self, windows_images=10, windows_returns=10):
        # merget the X and y by times column
        self._generate_ob_image(windows_images)
        self._generate_image_label(windows_returns)
        self.res = self.images_id_df.merge(self.labels_df, how="inner", on="times")
        return self.images[self.res["images_id"]], self.res["returns"].values.reshape(-1, 1)
    
    def get_times(self):
        return self.res["times"]
    
if __name__ == "__main__":
    data = pd.read_csv(r"data/order_book.csv")
    iob = IOBImage(data)
    X,y = iob.get_X_y()
