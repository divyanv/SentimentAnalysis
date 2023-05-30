# pip install scikit-learn

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('AmazonReview\Reviews.csv')

# Preprocess the data
data = data[['Text', 'Score']]  # Extract the relevant columns (Text and Score)
data = data.dropna()  # Remove any rows with missing values

# Map the scores to sentiment labels
data['Sentiment'] = data['Score'].apply(lambda score: 'positive' if score > 3 else 'negative' if score < 3 else 'neutral')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Sentiment'], test_size=0.2, random_state=42)

# Create a CountVectorizer for feature extraction
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(X_train_vectors, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)