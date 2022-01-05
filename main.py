import os
import time
import shutil
import datetime
import numpy as np
from model.NLTK import build_nltk_model, predict_nltk
from model.models import Randomforestclassifier, eval_rfc, eval_xgb, gridsearch_run, pre_processing, train_test
from utils.utils import triple_barrier
from utils.process_dataframe import pre_processing_csv
from utils.yahoo_finance import requests_url_price

from utils.metrics import get_eval_csv, evaluation_metrics


if __name__ == '__main__':
    output_path = './result/output'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # get stock from yahoo finance and labeled triple barrier
    time_list = ["2000-01-03 08:00:00", "2017-01-01 08:00:00"]
    period1_datetime = time_list[0]
    period2_datetime = time_list[1]
    period1 = int(time.mktime(datetime.datetime.strptime(period1_datetime, "%Y-%m-%d %H:%M:%S").timetuple()))
    period2 = int(time.mktime(datetime.datetime.strptime(period2_datetime, "%Y-%m-%d %H:%M:%S").timetuple()))  # 轉成字串
    all_price = requests_url_price(URL="https://finance.yahoo.com/quote/%5EDJI/history?period1=" + str(period1) + "&period2=" + str(period2) + "&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true")
    all_labeled = triple_barrier(all_price['close'], 1.04, 0.98, 20)
    all_labeled['date_string'] = all_price.date_time.dt.date.apply(lambda x: x.strftime('%Y-%m-%d'))
    all_labeled = all_labeled.set_index('date_string')
    # set triple barrier label
    train_csv, test_csv = get_eval_csv('./data/Stock-Sentiment-Analysis-main/Stock News Dataset.csv')
    prepro_headlines = pre_processing_csv(train_csv)
    prepro_headlines_test = pre_processing_csv(test_csv)
    train_csv['triple_Label'] = all_labeled['triple_barrier_signal'][train_csv['Date']].reset_index()['triple_barrier_signal']
    test_csv['triple_Label'] = all_labeled['triple_barrier_signal'][test_csv['Date']].reset_index()['triple_barrier_signal']

    # load nltk model
    vader = build_nltk_model()
    # load or train randomforest
    print('training RandomForestClassifier ...')
    randomclassifier, countvector = Randomforestclassifier(train_csv, prepro_headlines, model_name='./result/RF_model')
    print('finish!!')
    print('Running XGBoost ...')
    # load or train xgboost
    tain_data = pre_processing(prepro_headlines)
    test_data = pre_processing(prepro_headlines_test)
    # gridsearch_run(tain_data, train_csv['Label'])
    xgboostclassifier = train_test([[tain_data, train_csv['triple_Label']], [], [test_data, test_csv['triple_Label']]], model_name='./result/XGBoost')


    # Detect predictions
    df_predictions = predict_nltk(vader, test_csv.reset_index(), test_csv['Top25'].values)
    pred = df_predictions['compound'].values
    pred[pred > 0] = 1
    pred[pred <= 0] = 0

    pred_rfc = eval_rfc(randomclassifier, countvector, test_csv)
    pred_xgb = eval_xgb(xgboostclassifier, countvector, test_csv)


    # Evaluation model
    print('eval nltk by label')
    evaluation_metrics(test_csv['Label'].values, pred, output_name=output_path + '/eval.txt')
    print('eval randomforestclassifier by triple label')
    evaluation_metrics(test_csv['triple_Label'].values, pred_rfc, output_name=output_path + '/eval_triple_rfc.txt')
    pred_rfc[pred_rfc == 2] = 0
    print('eval randomforestclassifier by label')
    evaluation_metrics(test_csv['Label'].values, pred_rfc, output_name=output_path + '/eval_rfc.txt')
    pred_xgb = np.argmax(pred_xgb, axis=1)
    print('eval XGBoost by triple label')
    evaluation_metrics(test_csv['triple_Label'].values, pred_xgb, output_name=output_path + '/eval_triple_xgb.txt')

