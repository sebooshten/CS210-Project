import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as stlit
import string
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from collections import defaultdict
from operator import itemgetter
from textblob import TextBlob
import emoji


#This was the  attempt at trying to account for each and every glyph/symbol of each language
#As documented in the paper, it was near impossible to account for all this and I had to decide to 
#only include english for a realistic due date.

#Add all fonts into matplotlib so the glyphs and other languages can go through
#font_dir 'C:\\Users\\seb\\AppData\\Local\\Microsoft\\Windows\\Fonts'
#font_files  matplotlib.font_manager.findSystemFonts(fontpaths=[font_dir])

#for file in font_files
    #fm.fontManager.addfont(file)

#plt.rcParams['font.family']  'Noto Sans'


tweets_csv = pd.read_csv('/Users/seb/Documents/CS210/tweets.csv',engine ='python')

#clean tweets using re library
def clean(tweet):
    
    tweet = re.sub(r'http\S+', '', tweet) #greedy (+) modifer to take everything attached as "one" item, for \S any non-whitespace character
    tweet = re.sub(r'@\w+', '', tweet) #any alpha-numeric lettering for @ symbol which can take on a persons username (removes symbol and what comes after, used as person notifier)
    tweet = re.sub(r'@\s+', '', tweet) #any whitespace after @ symbol (will remove symbol but not what comes after, used as location marker)
    tweet = re.sub(r'#\w+', '', tweet) #any hashtag w alphanumeric after
    tweet = re.sub(r'#\s+', '', tweet) #any hashtag w whitespace after
    tweet = emoji.replace_emoji(tweet, replace='') #remove emojis
    tweet = contractions.fix(tweet) #handle contractions early to help with stemming
    tweet = tweet.translate(str.maketrans('', '', string.punctuation)) #removes all punctuation by mapping punctuation characters to None. Its telling you which ones you want to map to and from and which characters you want deleted, which in this case is punctuation.
    tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet) #Include only English
    return tweet

tweets_csv['cln_tweet'] = tweets_csv['content'].apply(clean) #take filters and make a new column w/ that applied to original col

#removing stopwords initialization (if, the , as, any...)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#stemming allows us to reduce a word down to its stem. Almost like instead of clipping off all the branches of a specific kind of tree, find its root then remove from their so all its branches come with it, instead of running, runs, they all break down to run.
stemmer = SnowballStemmer('english')
nltk.download('punkt')

#stem/tokenize
def tokenize(tweet):
    tokens = word_tokenize(tweet) #splits sentence into tokenized words
    tokens = [stemmer.stem(w) for w in tokens if w.lower() not in stop_words] #for each word in tokens, if the word(to lower) is in the stop_words dictionary, we do not want to take it. Then, stem each of those words
    return ' '.join(tokens) #fills empty string with tokens to create the sentence again 

tweets_csv['tknzd_tweet'] = tweets_csv['cln_tweet'].apply(tokenize)

#dictionary of most frequent words and matplotlib for plot
#using defaultdict instead of normal dictionary for simpler code as well as removes the need to check if key exists

tokenized_sentences = tweets_csv['tknzd_tweet'] #grab column containing tokenized words
frequency_dict = defaultdict(int) #int for default value for new keys

for sentence in tokenized_sentences:
    words = sentence.split()
    for word in words:
        frequency_dict[word] += 1
    

#simply plotting these words and frequencies is impossible since we are dealing with thousands of words
#best case is to just use the top 100 most frequent words

#must change dictionary into a list of tuples, that way we can use a sorting method based on the second object (the value/frequency)
frequency_tuples = list(frequency_dict.items())

#use itemgetter to sort by second object in tuple, and set that function as for what to be used for the "key" parameter for sort function, reverse order for high -> low
frequency_tuples.sort(key= itemgetter(1), reverse= True)

#80, since 100 was too much for graph
eighty_mostFrequent = frequency_tuples[:80]

#unzip the tuple list, to get all the x values(words) and y values(frequencies). in the case of my tuple (word, freq), zip* will take this list of tuples
#and return a separate tuple for every single "word" that was in the list of tuples, along with a separate tuple for their associated "frequencies" which has every single freq in the list of tuples

plot_words, plot_frequencies = zip(*eighty_mostFrequent)

#create plot with words/freq
plt.figure(figsize= (17,10))
plt.bar(plot_words, plot_frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 80 Most Frequent Words')
plt.xticks(rotation = 60) #to fit on screen
plt.tight_layout()
plt.show()


#sentiment analysis

def sentiment_analysis(tweet):
    polarity = TextBlob(tweet).polarity #grab polarity [-1,1] of tweet

    if polarity > 0 :
        return 'Positive'
    elif polarity == 0 :
        return 'Neutral'
    else:
        return 'Negative'

tweets_csv['sentiment'] = tweets_csv['tknzd_tweet'].apply(sentiment_analysis)

num_positive = tweets_csv['sentiment'].value_counts()['Positive']
num_neutral = tweets_csv['sentiment'].value_counts()['Neutral']
num_negative =tweets_csv['sentiment'].value_counts()['Negative']

print(f'\nNumber of Positive Tweets =  {num_positive} \nNumber of Negative Tweets =  {num_neutral} \nNumber of Neutral Tweets  =  {num_negative}')

#export to new csv
tweets_analysis = tweets_csv[['author', 'content', 'cln_tweet', 'tknzd_tweet', 'date_time', 'id', 'number_of_likes', 'number_of_shares', 'sentiment']]
tweets_analysis.to_csv('C:\\Users\\seb\\Documents\\CS210\\tweets_analysis.csv', index = False)

#streamlit dashboard for visualization

#Work is done in Separate File

#cmd line arguments for Streamlit Dashboard
#cd C:\Users\seb\Documents\CS210\Main Project
#streamlit run stlit_dashboard.py








