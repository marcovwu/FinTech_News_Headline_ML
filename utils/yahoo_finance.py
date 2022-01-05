import time
import requests
import pandas as pd
from ast import literal_eval


def requests_url_price(URL):  # , show=False, output_name='output/test.png'):
    res = requests.get(URL, headers={'User-agent': 'Mozilla/5.0'})

    #ast.literal_eval(+ '}')
    info_string = res.text.split('HistoricalPriceStore')[1].split('isPending')[0]
    info_list = info_string[info_string.find('['):info_string.find(']')][1:].replace('},{', '}@{').split('@')
    for date_info in info_list:
        if 'null' in date_info:
            period = literal_eval(date_info.replace('null', '"null"'))['date']
            print(period, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(period)), date_info)
    dataframe = pd.DataFrame([literal_eval(date_info.replace('null', '"null"')) if 'null' in date_info else literal_eval(date_info) for date_info in info_list])

    dataframe['date_time'] = pd.to_datetime(dataframe['date'], unit='s')
    # sort by date and time
    dataframe = dataframe.sort_values(by=['date'], ascending=True)
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe

