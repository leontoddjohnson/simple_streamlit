import re
import pandas as pd
import plotly.express as px


def get_sentiment_data(df, text_col, analyzer):
    # gather sentiment scores for data frame `df`
    df_sentiment = []

    for review in df[text_col]:
        vs = analyzer.polarity_scores(review)
        df_sentiment.append(vs)

    df_sentiment = pd.DataFrame(df_sentiment,
                                index = df.index)
    df_sentiment = pd.concat((df, df_sentiment), axis=1)

    return df_sentiment


def plot_sentiment(df_sentiment, benchmarks):
    df_plot = df_sentiment.melt(id_vars=['name', 'roaster'], 
                            value_vars=['neg', 'neu', 'pos', 'compound'],
                            var_name='sentiment_type', value_name='amount')

    fig = px.strip(df_plot, x='sentiment_type', y='amount', 
                template='simple_white', log_y=True,
                hover_name='name',
                hover_data=['roaster'])

    df_ = benchmarks.loc['mean']

    fig.add_scatter(x=df_.index, y=df_, 
                    mode="markers", marker_size=10, marker_color='darkorange', name='review_average')
    
    return fig


def get_sentence_sentiment(text, analyzer):
    sentences = re.split('[?.!]', text)
    sentences = [s for s in sentences if s != '']

    df_ = pd.DataFrame(sentences, 
                       columns=['text'])
    
    df_sentiment = get_sentiment_data(df_, 
                                    text_col='text', 
                                    analyzer=analyzer)
    
    return df_sentiment