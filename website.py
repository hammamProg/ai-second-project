from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords') 
stop_words = set(nltk.corpus.stopwords.words("arabic"))

app = Flask(__name__)

######## Load the vectorizer and classifiers #########
vectorizer = joblib.load('vectorizer.joblib')
dt = joblib.load('dt_model.joblib')
nb = joblib.load('nb_model.joblib')
rf = joblib.load('rf_model.joblib')
######################################################

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    cleaned_tweet = preprocess_tweet_text(tweet)
    # Vectorize the tweet
    vectorized_tweet = vectorizer.transform([cleaned_tweet])
    # Predict the sentiment using each classifier
    dt_prediction = dt.predict(vectorized_tweet)
    nb_prediction = nb.predict(vectorized_tweet)
    rf_prediction = rf.predict(vectorized_tweet)
    
    return render_template('result.html', tweet=tweet, dt_prediction=dt_prediction, nb_prediction=nb_prediction, rf_prediction=rf_prediction)

def preprocess_tweet_text(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stop_words]
    return ' '.join(filtered_words)

if __name__ == '__main__':
    app.run()
