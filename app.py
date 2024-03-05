import streamlit as st
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model
model = load_model("your_model.h5")  # Replace "your_model.h5" with the actual path to your saved model file

# Load the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(decode_error='replace', strip_accents='unicode', stop_words='english')

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    tweet_sentences = ' '.join(tweets_clean)

    return tweet_sentences

def predict(string):
    # Preprocess the input string
    string_clean = process_tweet(string)

    # Tokenize the cleaned string
    tokens = string_clean.split()

    # Initialize a list to store corrected tokens
    corrected_tokens = []

    for token in tokens:
        # Check if the token is a valid word
        if wordnet.synsets(token):
            corrected_tokens.append(token)
        else:
            # Find the most similar word for unidentified words
            suggestions = wordnet.synsets(token)
            if suggestions:
                most_similar_word = suggestions[0].lemmas()[0].name()
                corrected_tokens.append(most_similar_word)
            else:
                corrected_tokens.append(token)  # If no suitable replacement found, keep the original token

    # Join the corrected tokens into a string
    corrected_string = ' '.join(corrected_tokens)

    # Process the corrected string
    string_vectorized = tfidf_vectorizer.transform([corrected_string])

    # Sort the indices to ensure they are in order
    string_vectorized.sort_indices()

    # Make predictions
    predicted_probabilities = model.predict(string_vectorized)[0]  # Assuming predictions is an array with a single element

    # Get the predicted class
    predicted_class = np.argmax(predicted_probabilities)

    return predicted_class

# Streamlit app
def main():
    st.title("Text Classifier")
    input_text = st.text_input("Enter text:")
    if st.button("Predict"):
        prediction = predict(input_text)
        st.write("Predicted class:", prediction)

if __name__ == "__main__":
    main()
