from finvizfinance.quote import finvizfinance


def get_stocknews_finviz(name='tsla'):
    stock = finvizfinance(name)
    news_df = stock.TickerNews()

    return news_df

