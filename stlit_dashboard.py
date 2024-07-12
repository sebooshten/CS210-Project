import streamlit as stlit
import pandas as pd
import main 

tweets_analyze = pd.read_csv('/Users/seb/Documents/CS210/tweets_analysis.csv',engine ='python')

stlit.title('Sentiment Analysis Dashboard')

#Need counts inside of a dataframe to place in graph, counts have to be separate y-values for area chart to work
num_counts = pd.DataFrame({'Sentiment': ['Positive', 'Neutral', 'Negative'], 
                          'Positive Visual' :[main.num_positive, 0, 0],
                          'Neutral Visual' : [0, main.num_neutral, 0],
                          'Negative Visual' : [0, 0, main.num_negative]})

#create area chart
stlit.area_chart(num_counts.set_index('Sentiment') ,color = ['#dd696c','#a0a3a7', '#a4d17b' ], x_label = 'Sentiment', y_label = 'Count')

stlit.subheader('Sentiment Filter:')
sentiment_choice = stlit.selectbox('Choose Sentiment:', ['All', 'Positive', 'Neutral', 'Negative'])

#display based on choice
if sentiment_choice == 'All':
    tweets = tweets_analyze[['content', 'sentiment']]
else:
    tweets = tweets_analyze[tweets_analyze['sentiment'] == sentiment_choice]['content']

stlit.dataframe(tweets, use_container_width = True)