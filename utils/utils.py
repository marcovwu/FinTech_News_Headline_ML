import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def triple_barrier(price, ub, lb, max_period):
    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0] / s[0]

    r = np.array(range(max_period))

    def end_time(s):
        return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], max_period - 1)[0]

    p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period + 1)
    t = price.rolling(max_period).apply(end_time, raw=True).shift(-max_period + 1)
    t = pd.Series([t.index[int(k + i)] if not math.isnan(k + i) else np.datetime64('NaT')
                   for i, k in enumerate(t)], index=t.index).dropna()

    signal = pd.Series(0, p.index)
    signal.loc[p > ub] = 1
    signal.loc[p < lb] = 2
    ret = pd.DataFrame({'triple_barrier_profit': p, 'triple_barrier_sell_time': t, 'triple_barrier_signal': signal})

    return ret


def plot_title_bar(news, show=False, output_name='output/test.png'):
    mean_c = news.copy()
    plt.style.use("fivethirtyeight")
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_c['Date'] = mean_c['Date'].dt.date
    mean_c = mean_c.groupby(['Date', 'ticker']).mean()
    # Unstack the column ticker
    mean_c = mean_c.unstack(level=1)
    # Get the cross-section of compound in the 'columns' axis
    mean_c = mean_c.xs('compound', axis=1)
    # Plot a bar chart with pandas
    mean_c.plot.bar()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.ioff()
    plt.savefig(output_name)
    plt.close()

def plot_title_3bar(news, TITLE="Positive, negative and neutral sentiment for FB on 2021-12-03", show=False, output_name='output/test.png'):
    COLORS = ["red", "orange", "green"]
    # Drop the columns that aren't useful for the plot
    #plot_day = news.drop(['Title', 'compound'], axis=1)
    plot_news = news[['neg', 'pos', 'neu']]
    #plot_news.colume = ['negative', 'positive', 'neutral']
    # Plot a stacked bar chart
    plot_news.plot.bar(stacked=True,
                       figsize=(30, 15),
                       title=TITLE,
                       color=COLORS).legend(bbox_to_anchor=(1.05, 0.5))
    plt.ylabel("scores")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.ioff()
    plt.savefig(output_name)
    plt.close()


def plot_title_line(news, TITLE="Compound for FB on 2021-12-03", show=False, output_name='output/test.png'):
    COLORS = ["red", "orange", "green"]
    # Drop the columns that aren't useful for the plot
    plot_news = news[['compound']]
    # Plot a stacked bar chart
    plot_news.plot.line(figsize=(30, 15), title=TITLE, color=COLORS[0])
    plt.ylabel("scores")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.ioff()
    plt.savefig(output_name)
    plt.close()


### Plot results
def plot_train_results(progress, show=False, output_name='./train.png'):
    global pickle_validate

    eval_result = progress['eval']['mlogloss']
    train_result = progress['train']['mlogloss']
    x_range = list(range(1, len(progress['eval']['mlogloss']) + 1, 1))
    p1 = plt.plot(x_range, eval_result, c='blue', label='eval')
    p2 = plt.plot(x_range, train_result, c='orange', label='train')

    if len(progress) > 2:
        validate_result = progress['val']['mlogloss']
        p3 = plt.plot(x_range, validate_result, c='red', label='validate')

    plt.xlabel("Rounds")
    plt.ylabel("logloss")
    plt.legend(loc='upper right')
    if show:
        plt.show()
    else:
        plt.ioff()
    plt.savefig(output_name)
    plt.close()

import os
import time
import platform
from pathlib import Path
def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', file='tmp.zip'):
    # Downloads a file from Google Drive. from yolov5.utils.google_utils import *; gdrive_download()
    t = time.time()
    file = Path(file)
    cookie = Path('cookie')  # gdrive cookie
    print(f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ', end='')
    #file.unlink(missing_ok=True)  # remove existing file
    #cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists('cookie'):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    #cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == '.zip':
        print('unzipping... ', end='')
        os.system(f'jar xvf {file}')  # unzip
        file.unlink()  # remove zip to free space

    print(f'Done ({time.time() - t:.1f}s)')
    return r

def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""


if __name__ == '__main__':
    os.makedirs('./result', exist_ok=True)
    gdrive_download('1QFrP0iALlUeDIFnK5x1ZIZIhV3x-tshr', file='./result/RF_model')
    gdrive_download('1223-EoVkpSntDHrIigjryszO_jjKnHAv', file='./result/XGBoost')