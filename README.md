Combined NLP Projects README

This file contains the README documentation for two separate NLP projects:

Twitter Hate Speech Detection

Song Recommendation System

Project 1: Twitter Hate Speech Detection

This project uses Natural Language Processing (NLP) and a machine learning model to detect hate speech in tweets. The primary objective is to classify tweets as racist/sexist or not. The model is built using scikit-learn and nltk and relies on a Bag-of-Words (BoW) feature extraction method.

📜 Project Objective

The goal of this task is to identify tweets containing hate speech. For this project, a tweet is considered hate speech if it has racist or sexist sentiment.

The task is to build a classification model that, given a set of tweets, will predict one of two labels:

Label '1': The tweet is racist/sexist.

Label '0': The tweet is not racist/sexist.

💾 Dataset

The model was trained on a labeled dataset containing 31,962 tweets.

File: train_E60V31V.csv

Format: The CSV file contains three columns:

id: int64 - A unique identifier for each tweet.

label: int64 - The sentiment label (0 or 1).

tweet: object - The raw text of the tweet.

🛠️ Project Pipeline

The project follows a standard NLP workflow: Data Preprocessing, Exploratory Data Analysis (EDA), Feature Extraction, and Model Training.

1. Data Preprocessing

The raw tweet text (tweet column) was cleaned to create a new clean_tweet column. This involved several steps:

Remove Twitter Handles: Removed all Twitter handles (e.g., @user) using a custom remove_pattern function with the regex @[\w]*.

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")


Remove Special Characters: Removed all special characters, numbers, and punctuation, except for the hashtag symbol (#).

df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")


Remove Short Words: Words with a length of 3 or less were removed (e.g., "so", "is").

Tokenization: Each cleaned tweet was split into a list of individual words (tokens).

tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())


Stemming: Tokens were reduced to their root form using the PorterStemmer from nltk. (e.g., "dysfunctional" becomes "dysfunct").

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])


Rejoin Tokens: The list of stemmed tokens was joined back into a single string for each tweet.

2. Exploratory Data Analysis (EDA)

EDA was performed to find insights in the cleaned data:

Word Clouds: Word clouds were generated to visualize the most frequent words in:

All tweets combined.

Non-hate tweets (label 0).

Hate-speech tweets (label 1).
Words like love, happi, and bihday were common in positive tweets, while trump, white, black, libtard, and allahsoil were prominent in negative tweets.

Hashtag Analysis: Hashtags were extracted from all tweets using the regex r"#(\w+)".

The top 10 most frequent hashtags from non-hate tweets (label 0) were plotted. Top 3: #love, #posit, #smile.

The top 10 most frequent hashtags from hate-speech tweets (label 1) were plotted. Top 3: #trump, #polit, #allahsoil.

3. Feature Extraction

The cleaned, stemmed text was converted into a numerical format suitable for modeling.

Technique: Bag-of-Words (BoW)

Implementation: sklearn.feature_extraction.text.CountVectorizer.

Parameters:

max_df=0.90 (Ignore terms that appear in > 90% of documents).

min_df=2 (Ignore terms that appear in < 2 documents).

max_features=1000 (Use only the top 1000 most frequent terms).

stop_words='english' (Remove common English stop words).

4. Model Building & Evaluation

Train-Test Split: The BoW features (bow) and the target (df['label']) were split into training and testing sets using train_test_split.

Model Selection: A Logistic Regression model was chosen for classification.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)


Evaluation: The model was evaluated using f1_score (crucial for imbalanced datasets) and accuracy_score.

📊 Results

The model's performance was evaluated in two stages:

1. Default Model (Using .predict()):

F1-Score: 0.508

Accuracy: 0.948

2. Tuned Model (Adjusting Probability Threshold):
By adjusting the prediction threshold for the positive class, the F1-score was improved.

F1-Score: 0.561

Accuracy: 0.944

This shows that tuning the probability threshold for the positive class (label 1) can significantly improve the F1-score, which is a better measure of success for this type of imbalanced classification problem.

📦 Core Dependencies

This project relies on the following Python libraries:

Data Manipulation: pandas, numpy

Data Visualization: matplotlib, seaborn

Text Processing: re, string, nltk

NLP/ML: wordcloud, scikit-learn (sklearn)

Utilities: warnings

Project 2: Song Recommendation System

This project is a content-based song recommendation system. It uses the lyrics of songs to find and recommend other songs with similar lyrical content.

The model is built by processing song lyrics through natural language processing (NLP) techniques, converting them into numerical vectors using TF-IDF, and then calculating the cosine similarity between all songs.

💾 Dataset

Source: spotify_millsongdata.csv

Original Size: 57,650 songs

Columns: artist, song, link, text

Note: For performance and demonstration purposes, this project works with a random sample of 5,000 songs from the original dataset. The link column is also dropped during this process.

🛠️ Project Pipeline

The recommendation system is built in the following steps:

1. Data Loading & Sampling

The spotify_millsongdata.csv file is loaded into a pandas DataFrame. A random sample of 5,000 songs is taken to create a smaller, more manageable dataset.

import pandas as pd
df = pd.read_csv("spotify_millsongdata.csv")
df = df.sample(5000).drop('link', axis=1).reset_index(drop=True)


2. Text Preprocessing (NLP)

To make the lyric text usable for the model, it is cleaned and processed:

Lowercase: All text is converted to lowercase.

Remove Newlines: Newline characters (\n) are removed.

Tokenization & Stemming: A custom tokenization function is applied to each song's lyrics:

The text is tokenized (split into individual words) using nltk.word_tokenize.

Each word is stemmed (reduced to its root form, e.g., "wandering" -> "wander") using nltk.stem.porter.PorterStemmer.

The stemmed words are joined back into a single string.

import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

df['text'] = df['text'].apply(lambda x: tokenization(x))


3. Feature Extraction (TF-IDF)

The cleaned, stemmed lyrics are converted into a numerical matrix using TF-IDF (Term Frequency-Inverse Document Frequency). This method vectorizes the text, giving more weight to words that are important to a specific song rather than words that appear in all songs.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(df['text'])


4. Similarity Calculation

Cosine Similarity is used to calculate the similarity between all song vectors in the TF-IDF matrix. This results in a 5000x5000 similarity matrix where each cell (i, j) represents the similarity score between song i and song j.

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(matrix)


🚀 How to Use

A recommendation function is used to find similar songs.

Function: recommendation(song_df)

Input: The title of a song (e.g., "I'm Yours").

Process:

It finds the index of the input song in the DataFrame.

It retrieves that song's similarity scores from the similarity matrix.

It sorts the scores in descending order and finds the top 20 most similar songs (skipping the first result, which is the song itself).

Output: A list of the top 20 most similar song titles.

def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x:x[1])
    
    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].song)
    return songs


Example Usage:

recommendation("I'm Yours")


Example Output:

["I'm Yours", 'Love To Love', 'High Steppin' Proud', 'Tomorrow', 'Come Tomorrow', 'Alone Tonight', 'Borrowed Time', 'I Have Dreamed', 'More Than I Can Say', 'Make Tomorrow', 'Help Me Make It Through The Night', 'Feels Like Heaven', 'Why Do I Love You?', 'My Kind Of Lady']


📦 Final Artifacts

The script saves the two key components required for the recommendation system to run without retraining:

similarity.pkl: The 5000x5000 cosine similarity matrix.

df.pkl: The 5000-row pandas DataFrame containing the song information and processed text.

import pickle
pickle.dump(similarity, open('similarity.pkl','wb'))
pickle.dump(df, open('df.pkl', 'wb'))


📚 Core Libraries Used

pandas

nltk

scikit-learn (sklearn)

pickle
