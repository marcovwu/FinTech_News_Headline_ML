def make_times_dataframe(dataframe, companies='tsla', start_time='2021-12-03 00:00:00', end_time='2021-12-04 00:00:00'):
    # select single time
    single_time = dataframe[(dataframe['Date'] > start_time) & (dataframe['Date'] < end_time)]
    # loc campanies
    single_info = single_time.set_index(['ticker']).loc[companies]
    # Sort it
    # single_time['Date'].dt.time
    single_info = single_info.set_index('Date').sort_index(ascending=True)

    return single_info


def pre_processing_csv(dataframe):
    data = dataframe.iloc[:, 2:27]
    data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    # Renaming column names for better understanding and ease of access
    new_Index = [str(i) for i in range(25)]
    data.columns = new_Index

    # converting headlines to lower case
    for index in new_Index:
        data[index] = data[index].str.lower()

    # merge all top titles
    headlines = []
    for row in range(0, len(data.index)):
        headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))

    return headlines