import pandas as pd


def cal_VWAP(dataframe, name='means'):
    dataframe[name] = dataframe['compound'].mean()
    return dataframe

def add_means(dataframe, period):
    time_group = dataframe.groupby(pd.Grouper(key='Date', freq=period))
    dataframe = time_group.apply(cal_VWAP, name='means')
    return dataframe