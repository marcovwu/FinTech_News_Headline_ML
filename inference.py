import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from model.models import predict_xgb_period, build_xgb
from model.NLTK import build_nltk_model, predict_nltk
from utils.finviz import get_stocknews_finviz
from utils.utils import plot_title_bar, plot_title_3bar, plot_title_line
from utils.process_dataframe import make_times_dataframe
from utils.post_dataframe import add_means


def inference(vader, companies, DATE_TIME, MODE='nltk', ori_path='./synthesis/', ori_output_path='./result/', old_news=False, new_news=False):
    # Get News Title
    pbar = tqdm(enumerate(companies), total=len(companies))
    for i, company in pbar:
        output_path = ori_output_path + company
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        if new_news:
            News = get_stocknews_finviz(company)
            ticker_list = [company for _ in range(News.shape[0])]
            ticker = pd.DataFrame({'ticker': ticker_list})
            News = pd.concat([ticker, News], axis=1)
        if old_news:
            assert os.path.exists(ori_path + company + '.csv'), "No such file : %s if inference MODE 0 in the first time, you need to set old_path = './synthesis/'. if inference MODE 1 you need to have file in --old_path {your_file_path}" % ( ori_path + company + '.csv')
            ori_News = pd.read_csv(ori_path + company + '.csv', encoding='utf-8')
            ori_News = ori_News.drop(['means', 'pos', 'neg', 'compound'], axis=1)
            if new_news :
                News = pd.concat([ori_News, News], axis=0).drop_duplicates(subset='Title').reset_index(drop=True)
            else:
                News = ori_News
        News['Date'] = pd.to_datetime(News['Date'], format="%Y-%m-%d %H:%M:%S")
        News = News.sort_values(by=['Date'], ascending=True, ignore_index=True)
        # set News  in DATE_TIME[0] > DATE_TIME[1]
        News = News[News['Date'] < DATE_TIME[1]]
        News = News[News['Date'] < DATE_TIME[1]]
        if len(News):
            # inference
            if MODE == 'nltk':
                News = predict_nltk(vader, News, News['Title'].values)
                output_News = add_means(News, period='1D')
            elif MODE == 'xgb':
                output_News = predict_xgb_period(vader[0], vader[1], News, period='1D', DATA_NAME='Title')

            single_news = make_times_dataframe(output_News, companies=[company], start_time=opt.DATE_TIME[0], end_time=DATE_TIME[1])
            # write csv
            output_News.to_csv(output_path + '.csv', index=False, encoding='utf-8')

            # Plot result
            plot_title_bar(output_News, output_name=output_path + '/' + company + '_title_bar.png')
            plot_title_3bar(single_news, TITLE="Positive, negative and neutral sentiment for " + company + " on " + DATE_TIME[0].split(' ')[0] + " to " + DATE_TIME[1].split(' ')[0], output_name=output_path + '/' + company + '_title_3bar.png')
            plot_title_line(single_news, TITLE="Compound for " + company + " " + DATE_TIME[0].split(' ')[0] + " to " + DATE_TIME[1].split(' ')[0], output_name=output_path + '/' + company + '_title_line.png')


def synthesis_output(vader, companies, DATE_TIME, MODE, ori_path='./synthesis/', ori_output_path='./result/synthesis/'):
    if not os.path.exists(ori_output_path):
        os.makedirs(ori_output_path, exist_ok=True)
        print('Start get News ...')
        inference(vader, companies, DATE_TIME, MODE=MODE, ori_output_path=ori_output_path, new_news=True)
    else:
        print('Synthesis old and new title ...')
        inference(vader, companies, DATE_TIME, MODE=MODE, ori_path=ori_path, ori_output_path=ori_output_path, old_news=True, new_news=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('--companies', nargs='+', type=str, default=['nvda', 'tsla', 'aapl', 'amzn', 'fb'], help='')
    parser.add_argument('--model_mode', type=str, default='nltk', help='nltk or xgb model')
    parser.add_argument('--old_path', type=str, default='./result/synthesis/', help='')
    parser.add_argument('--output_path', type=str, default='./result/inference_output/', help='')
    parser.add_argument('--DATE_TIME', nargs='+', type=str, default=['2021-12-01 00:00:00', '2022-01-15 00:00:00'], help='will set output in date_time[0] to date_time[1]')
    parser.add_argument('--MODE', type=str, default='0', help='0: synthesis old and new news, 1: only old news inference, 2: only new news inference')
    opt = parser.parse_args()

    # Build model
    if opt.model_mode == 'xgb':
        vader = build_xgb('XGBoost_3')
    elif opt.model_mode == 'nltk':
        vader = build_nltk_model()

    # inference
    if opt.MODE == '0':
        synthesis_output(vader, opt.companies, opt.DATE_TIME, MODE=opt.model_mode, ori_path=opt.old_path, ori_output_path=opt.output_path)
    elif opt.MODE == '1':
        if os.path.exists(opt.output_path):
            shutil.rmtree(opt.output_path)
        inference(vader, opt.companies, opt.DATE_TIME, MODE=opt.model_mode, ori_path=opt.old_path, ori_output_path=opt.output_path, old_news=True)
    elif opt.MODE == '2':
        if os.path.exists(opt.output_path):
            shutil.rmtree(opt.output_path)
        os.makedirs(opt.output_path, exist_ok=True)
        inference(vader, opt.companies, opt.DATE_TIME, MODE=opt.model_mode, ori_output_path=opt.output_path, new_news=True)
