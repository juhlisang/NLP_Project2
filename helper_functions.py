import pandas as pd  
import collections
import numpy as np
from textblob import TextBlob # pip install -U textblob, python -m textblob.download_corpora
from textblob.classifiers import NaiveBayesClassifier
import nltk # pip install nltk, python -m textblob.download_corpora
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline # pip install -q transformers
import matplotlib.pyplot as plt

def getSortedLocationScores(data):
    location_scores = {}
    location_counts = collections.Counter(data['location'])
    for (location, count) in location_counts.items():
        score = locationSentiment(data, location)
        location_scores[location] = score

    # code sourced from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    sorted_location_scores = {k: v for k, v in sorted(location_scores.items(), key=lambda item: item[1])}
    return sorted_location_scores

def locationSentiment(data, location):
    """
    input: location is the input location of interest
    output: the average sentiment of the tweets for a location
    """
    # go through the dataframe filtered by location
    # get the average sentiment of the tweets in that location by getting sum and dividing by number of rows
    location_data = data[data["location"] == location]
    total_sentiment_value = sum(list(location_data["target"]))
    return total_sentiment_value/len(location_data)

def getLocationTweetsWithSentiment(data, location):
    tweets_with_sentiment = []
    location_data = data[data["location"] == location]
    for row in location_data.iterrows():
        sentiment = row[1]['target']
        text = row[1]['text']
        sentiment_for_classifier = ""
        if sentiment == 0:
            # negative
            sentiment_for_classifier = "neg"
        elif sentiment == 4:
            # positive
            sentiment_for_classifier = "pos"
        tweets_with_sentiment.append((text,sentiment_for_classifier))
    return tweets_with_sentiment

def averageSentimentByLocation(location, data):
    dataset = getLocationTweetsWithSentiment(data, location)
    train = dataset[:int(0.8*len(dataset))] # change this so that its actually a train test split 
    test = dataset[int(0.8*len(dataset)):]
    # naive bayes
    naive_bayes = NaiveBayesClassifier(train)
    total_sentiment = 0
    true_sentiment = 0
    for text, sentiment in test:
        pred = naive_bayes.classify(text)
        if sentiment == "pos":
            true_sentiment += 4
        if pred == "pos":
            total_sentiment += 4
        if pred == "neu":
            total_sentiment += 2
        # negative is still just value 0
    average_sentiment = total_sentiment/len(test)
    average_true = true_sentiment/len(test)
    return average_sentiment

def averageSentimentByLocation_sia(location, data):
    dataset = getLocationTweetsWithSentiment(data, location)
    test = dataset[int(0.8*len(dataset)):]
    sia = SentimentIntensityAnalyzer()
    total_sentiment = 0
    # using test so we can compare scores across models
    for text, _ in test:
        sia_pred = sia.polarity_scores(text)
        if sia_pred["compound"] > 0:
            total_sentiment += 4
        elif sia_pred["compound"] == 0:
            total_sentiment += 2
    average_sentiment = total_sentiment/len(dataset)
    return average_sentiment

def averageSentimentByLocation_hf(location, data):
    dataset = getLocationTweetsWithSentiment(data, location)
    test = dataset[int(0.8*len(dataset)):]
    specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    # note: model can only take 204 or 128 items in dataset, so only using test
    hf_test = test[:128]
    data = [text for text, sentiment in hf_test]
    results = specific_model(data)

    total_sentiment = 0
    for r in results:
        label = r['label'].lower()
        if label == 'pos':
            total_sentiment += 4
        if label == 'neu':
            total_sentiment += 2
    average_sentiment = total_sentiment/len(hf_test)
    return average_sentiment

def plot_grouped_bar(locations, actual_sentiments, predicted_sentiments, model):
    actual_sentiments_list = list(actual_sentiments.values())
    predicted_sentiments_list = list(predicted_sentiments.values())

    # Set the width of the bars
    bar_width = 0.35

    # Set the position of the bars on the x-axis
    r1 = np.arange(len(locations))
    r2 = [x + bar_width for x in r1]

    # Plotting
    plt.bar(r1, actual_sentiments_list, color='skyblue', width=bar_width, label='Actual Sentiment')
    plt.bar(r2, predicted_sentiments_list, color='orange', width=bar_width, label='Predicted Sentiment')

    # Add xticks on the middle of the group bars
    plt.xlabel('Location', fontweight='bold')
    plt.ylabel('Sentiment', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(locations))], locations, rotation=45, ha='right')

    # Create legend & Show graphic
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    plt.title(f'Actual vs Predicted Sentiment by Location for {model} model')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()