import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score
from utils.utils import plot_train_results
from utils.metrics import get_eval_csv, get_compound
from utils.process_dataframe import pre_processing_csv


def pre_processing(headline):
    countvector = CountVectorizer(ngram_range=(2, 2))
    data = countvector.fit_transform(headline)  ## implement RandomForest Classifier
    return data


def Randomforestclassifier(train, headlines, model_name='RF_model'):
    countvector = CountVectorizer(ngram_range=(2, 2))
    traindataset = countvector.fit_transform(headlines)  ## implement RandomForest Classifier

    if os.path.exists(model_name):
        randomclassifier = joblib.load(model_name)
    else:
        randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
        randomclassifier.fit(traindataset, train['Label'])
        joblib.dump(randomclassifier, model_name)

    return randomclassifier, countvector


def eval_rfc(randomclassifier, countvector, test):
    test_transform = []
    for row in range(0, len(test.index)):
        test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
    test_dataset = countvector.transform(test_transform)
    predictions = randomclassifier.predict(test_dataset)

    return predictions


def eval_xgb(xgboostclassifier, countvector, test):
    test_transform = []
    for row in range(0, len(test.index)):
        test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
    test_dataset = countvector.transform(test_transform)
    predictions = np.array(xgboostclassifier.predict(xgb.DMatrix(test_dataset)))

    return predictions


def build_xgb(model_path, path='/home/Marco/Marco/FinTech/Final_project_FinTech/datasets/Stock-Sentiment-Analysis-main/Stock News Dataset.csv'):
    prepro_headlines = pre_processing_csv(get_eval_csv(path)[0])
    countvector = CountVectorizer(ngram_range=(2, 2))
    countvector.fit_transform(prepro_headlines)
    xg_reg = joblib.load(model_path)
    vader = [xg_reg, countvector]
    return vader


def cal_predict_means(dataframe, xgb_model, countvector, data_name, name='means'):
    test_dataset = countvector.transform([' '.join(list(dataframe[data_name].values))])
    prediction = np.array(xgb_model.predict(xgb.DMatrix(test_dataset)))
    dataframe['neg'] = prediction[:, 2][0].mean()
    dataframe['pos'] = prediction[:, 1][0].mean()
    dataframe['neu'] = prediction[:, 0][0].mean()
    dataframe = get_compound(dataframe)
    dataframe[name] = dataframe['compound'].mean()
    return dataframe

def predict_xgb_period(xgboostclassifier, countvector, dataframe, period, DATA_NAME='Title'):
    time_group = dataframe.groupby(pd.Grouper(key='Date', freq=period))
    dataframe = time_group.apply(cal_predict_means, xgb_model=xgboostclassifier, countvector=countvector, data_name=DATA_NAME, name='means')
    return dataframe


### Set optimisation params for gridsearch here
# Tuning sequence see https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def set_gridsearch_params():
    params = {
        'n_estimators': [30],
        'max_depth': [9],
        'min_child_weight': [1],
        'gamma': [0.0],
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        # 'subsample':[0.95],
        # 'colsample_bytree':[0.95],
    }
    return params


### Gridsearch
def gridsearch_run(X_train, y_train):
    # Default classified which will be tuned
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        min_child_weight=1,
        gamma=0,
        subsample=0.5,
        colsample_bytree=0.5,
        learning_rate=0.1,  # ok for Gridsearch
        objective='multi:softprob',
        silent=True,
        nthread=1,
        num_class=2
    )

    # A parameter grid for XGBoost
    params = set_gridsearch_params()

    clf = GridSearchCV(xgb_model,
                       params,
                       cv=list(KFold(n_splits=5, shuffle=True).split(X_train)),  # at least 5 splits
                       verbose=2,
                       scoring='neg_log_loss',
                       n_jobs=-1
                       )

    grid_result = clf.fit(X_train, y_train.values.ravel())

    print("\n\nBest score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print("\nStats:")
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


### Set the best parameters here when step 2 is finished
def set_best_params():
    best_params = {
        'max_depth': 9,
        'min_child_weight': 5,
        'gamma': 0.0,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.001,
        'learning_rate': 0.01,  # fixed
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 3  # the number of classes that exist in this datset
    }
    return best_params


### Train - test and save
def train_test(data, model_name='XGBoost'):

    # Using the result
    boost_rounds = 1000  # this is ok
    early_stopping = 200  # tune this
    params = set_best_params()

    dtrain = xgb.DMatrix(data[0][0], label=data[0][1])
    dtest = xgb.DMatrix(data[2][0], label=data[2][1])
    if len(data[1]) == 0:
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    else:
        dval = xgb.DMatrix(data[1][0], label=data[1][1])
        watchlist = [(dtest, 'eval'), (dtrain, 'train'), (dval, 'validation')]

    if os.path.exists(model_name):
        xg_reg = joblib.load(model_name)
    else:
        progress = dict()

        # Train and predict with early stopping
        xg_reg = xgb.train(
            params=params,
            dtrain=dtrain, num_boost_round=boost_rounds,
            evals=watchlist,
            # using validation on a test set for early stopping; ideally should be a separate validation set
            early_stopping_rounds=early_stopping,
            evals_result=progress)
        # Save the model
        joblib.dump(xg_reg, model_name)
        # Plots
        plot_train_results(progress)

    ypred = np.array(xg_reg.predict(dtest))
    ypred_transformed = np.argmax(ypred, axis=1)

    # print ypred_transformed
    # print y_test.values.ravel()

    print('Precision', precision_score(data[2][1], ypred_transformed, average=None))
    print('F1', f1_score(data[2][1], ypred_transformed, average=None))
    importance = xg_reg.get_score(importance_type='gain')

    print('Feature importance')
    #for elem in importance:
        #print(elem)

    return xg_reg