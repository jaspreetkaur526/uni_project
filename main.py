import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from pandas import Index
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from wordcloud import WordCloud
from nltk.corpus import stopwords

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# removes pattern in the umput text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

# Function to create word cloud
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(
        text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

def get_sentiment(text):
    analysis = TextBlob(text)
    # Classify the polarity of the tweet
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

SOCIAL_df = pd.read_csv('Twitter_Sentiments.csv')
SOCIAL_df.columns
Index(['id', 'label', 'tweet'], dtype='object')
print('Length of data is ', len(SOCIAL_df))

SOCIAL_df.shape
SOCIAL_df.info()

# Explore the dataset
print('social df head')
print(SOCIAL_df.head())
print('dtypes')
print(SOCIAL_df.dtypes)

np.sum(SOCIAL_df.isnull().any(axis=1))
print(np.sum(SOCIAL_df.isnull().any(axis=1)))
print('Count of columns in the data is:  ', len(SOCIAL_df.columns))
print('Count of rows in the data is:  ', len(SOCIAL_df))

import nltk
# Download the stopwords dataset
nltk.download('stopwords')

print('processed tweets')
SOCIAL_df['processed_tweet'] = np.vectorize(remove_pattern)(SOCIAL_df['tweet'],"@[\w]*")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    SOCIAL_df['processed_tweet'], SOCIAL_df['label'], test_size=0.2, random_state=42
)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

classifier = make_pipeline(CountVectorizer(), MultinomialNB())
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Create a word cloud for processed tweets
processed_tweets_text = ' '.join(SOCIAL_df['processed_tweet'].values)
print('Creating word cloud')
create_wordcloud(processed_tweets_text)

# Visualize social interactions
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=SOCIAL_df)
plt.title('Distribution of Social Interactions')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Assuming SOCIAL_df is already loaded and has the 'processed_tweet' column
SOCIAL_df['processed_tweet_length'] = SOCIAL_df['processed_tweet'].apply(len)

# Visualization 1: Distribution of tweet lengths
plt.figure(figsize=(10, 6))
sns.histplot(SOCIAL_df['processed_tweet_length'], bins=30, kde=True)
plt.title('Distribution of Processed Tweet Lengths')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.show()

# Apply sentiment analysis to the 'processed_tweet' column
SOCIAL_df['sentiment'] = SOCIAL_df['processed_tweet'].apply(get_sentiment)

# Visualize the distribution of sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=SOCIAL_df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Perform sentiment analysis using TextBlob
SOCIAL_df['sentiment'] = SOCIAL_df['processed_tweet'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Visualize sentiment distribution
plt.figure(figsize=(8, 6))
SOCIAL_df['sentiment'].plot(kind='hist', bins=50, edgecolor='black', alpha=0.7)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# Word cloud visualization for positive and negative tweets
positive_tweets = SOCIAL_df[SOCIAL_df['sentiment'] > 0]['processed_tweet'].str.cat(sep=' ')
negative_tweets = SOCIAL_df[SOCIAL_df['sentiment'] < 0]['processed_tweet'].str.cat(sep=' ')

# Generate word clouds
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_tweets)

# Plot the word clouds
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Tweets Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Tweets Word Cloud')
plt.axis('off')

plt.show()

# Clustering using TF-IDF and KMeans
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(SOCIAL_df['processed_tweet'])

# Apply KMeans clustering
num_clusters = 5  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix)

# Visualize clustering results using PCA
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(tfidf_matrix.toarray())
SOCIAL_df['cluster'] = kmeans.labels_

# Scatter plot for clustered data
plt.figure(figsize=(10, 8))
for cluster in range(num_clusters):
    cluster_data = transformed_data[SOCIAL_df['cluster'] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')

plt.title('Clustering of Tweets')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()