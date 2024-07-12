import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
nltk.download('punkt')

#removing stopwords initialization (if, the , as, any...)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


#stem/tokenize
def tokenize(word):
    word = "Hello MY NAME IS george and i love to dance is because running is so much run to do fairly crazy"
    tokens = word_tokenize(word)
    print(tokens)

    tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
    print(tokens)

tokenize("hi")