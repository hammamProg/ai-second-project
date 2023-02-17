import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib import pyplot as plt  
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import joblib
import nltk
import time

# start timing
start_time = time.time()

nltk.download('punkt')
# Load the dataset
pos_tweets = pd.read_csv('positive_tweets.tsv', delimiter='\t', header=None, names=['label', 'tweet'])
neg_tweets = pd.read_csv('negative_tweets.tsv', delimiter='\t', header=None, names=['label', 'tweet'])

data = pd.concat([pos_tweets, neg_tweets], axis=0)

# Data preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('arabic'))


def preprocess_tweet_text(tweet):
    tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stop_words]
    return ' '.join(filtered_words)

data.tweet = data.tweet.apply(preprocess_tweet_text)

# function to extract features from the text
def extract_features(text):
    length_before = len(text)
    num_words_before = len(text.split())
    num_urls = len(re.findall("(http[s]?://)?([\w-]+\.)+([a-z]{2,5})(/+\w+)?", text))
    num_hashtags = len(re.findall(r'#(\w+)', text))
    num_mentions = len(re.findall(r'@(\w+)', text))
    text = preprocess_tweet_text(text)
    num_words_after = len(text.split())
    # create length_after feature
    length_after = len(text)
    return length_before, length_after, num_words_before, num_words_after, num_urls, num_hashtags, num_mentions

# Feature extraction
cv = CountVectorizer()
X = cv.fit_transform(data.tweet)
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X)
y = data.label

# Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
nb = MultinomialNB()
nb.fit(X_train, y_train)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Evaluate the classifiers using 5-fold cross-validation
dt_scores = cross_val_score(dt, X, y, cv=5)
nb_scores = cross_val_score(nb, X, y, cv=5)
rf_scores = cross_val_score(rf, X, y, cv=5)

print("Decision Tree Classifier Using 5-fold cross-validation:")
print("Accuracy: %0.3f (+/- %0.3f)" % (dt_scores.mean(), dt_scores.std() * 2))
print("\n")
print("Naive Bayes Classifier:")
print("Accuracy: %0.3f (+/- %0.3f)" % (nb_scores.mean(), nb_scores.std() * 2))
print("\n")
print("Random Forest Classifier:")
print("Accuracy: %0.3f (+/- %0.3f)" % (rf_scores.mean(), rf_scores.std() * 2))
print("\n")


# Save the trained models to disk
joblib.dump(dt, 'dt_model.joblib')
joblib.dump(nb, 'nb_model.joblib')
joblib.dump(rf, 'rf_model.joblib')
joblib.dump(cv, 'vectorizer.joblib')


# stop timing
end_time = time.time()
elapsed_time = end_time - start_time
# print the elapsed time
print(f"Time taken: {elapsed_time:.2f} seconds")
