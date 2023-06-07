import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import gensim
import sys

# Load the Word2Vec model
word2vec_model = gensim.models.Word2Vec.load("word2vec_model.bin")


# Load the sentiment analysis model
model = load_model('sentiment_analysis_model.h5')

# Helper functions to preprocess the data
def preprocess_text(text):
    # Tokenization and lowercasing
    text = word_tokenize(str(text).lower())

    # Removing special characters and numbers
    text = [word for word in text if word.isalpha()]

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]

    return text

def process_text(text):
    # Preprocess the text
    text = preprocess_text(text)

    # Handling rare words
    word_frequencies = nltk.FreqDist(text)
    rare_words = set(word for word in word_frequencies if word_frequencies[word] < 5)
    text = [word if word not in rare_words else 'UNK' for word in text]

    # Padding sequences (assuming a maximum sequence length of 100)
    max_seq_length = 100
    text = text[:max_seq_length] + ['PAD'] * (max_seq_length - len(text))

    return text

def predict_sentiment(text):
    # Preprocess the text
    processed_text = process_text(text)

    # Convert the text sequence to numerical sequence
    numerical_sequence = [word2vec_model.wv.key_to_index[word] if word in word2vec_model.wv.key_to_index else 0 for word in processed_text]

    # Pad the sequence
    padded_sequence = pad_sequences([numerical_sequence], maxlen=100, padding='post')

    # Make the prediction
    prediction = model.predict(padded_sequence)
    sentiment_label = np.argmax(prediction)

    # Map the sentiment label to the corresponding sentiment
    sentiments = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiments[sentiment_label]


    return sentiment


# Check if the script is run from the command line
if __name__ == "__main__":
    # Check if the command line arguments are provided
    if len(sys.argv) > 1:
        # Get the input text from the command line argument
        input_text = ' '.join(sys.argv[1:])
        # Call the predict_sentiment function with the input text
        print("Prediction : ",predict_sentiment(input_text))
    else:
        print("Please provide an input text.")

