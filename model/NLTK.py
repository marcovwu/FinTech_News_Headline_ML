# NLTK VADER for sentiment analysis
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# New words and values

def build_nltk_model():
    new_words = {
        'crushes': 10,
        'beats': 5,
        'misses': -5,
        'trouble': -10,
        'falls': -100,
    }
    # Instantiate the sentiment intensity analyzer with the existing lexicon
    vader = SentimentIntensityAnalyzer()
    # Update the lexicon
    vader.lexicon.update(new_words)
    return vader


def predict_nltk(vader, dataframe, titles, replace=False):
    scores = [vader.polarity_scores(sentence) for sentence in titles]
    # Convert the list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames
    if replace:
        dataframe[scores_df.columns] = scores_df
    else:
        dataframe = pd.concat([dataframe, scores_df], axis=1)
    # Convert the date column from string to datetime
    #scored_news['date'] = pd.to_datetime(scored_news.date).dt.date
    return dataframe