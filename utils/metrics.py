import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


def get_eval_csv(path_name):
    df = pd.read_csv(path_name, encoding="ISO-8859-1")
    train = df[df['Date'] < '20150101']
    test = df[df['Date'] > '20141231'].reset_index(drop=True)

    return train, test


def evaluation_metrics(predictions, label, output_name='output/test.txt'):
    matrix = confusion_matrix(label, predictions)
    print(matrix)
    score = accuracy_score(label, predictions)
    print(score)
    report = classification_report(label, predictions)
    print(report)

    with open(output_name, 'w+') as f:
        f.write('Matrix: ' + str(matrix) + '\n')
        f.write('Score: ' + str(score) + '\n')
        f.write(report)

    return score


def get_compound(dataframe):
    values = np.max([dataframe['pos'].values, dataframe['neg'].values, dataframe['neu'].values], axis=0)
    index = np.argmax([dataframe['pos'].values, dataframe['neg'].values, dataframe['neu'].values], axis=0)
    values[index == 1] = -values[index == 1]
    values[index == 2] = 0.0
    dataframe['compound'] = values

    return dataframe