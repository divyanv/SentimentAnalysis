import pandas as pd

# Load the dataset
data = pd.read_csv('AmazonReview\Reviews.csv')

# Preprocess the data
data = data[['Text', 'Score']]  # Extract the relevant columns (Text and Score)
data = data.dropna()  # Remove any rows with missing values

# Map the scores to sentiment labels
data['Sentiment'] = data['Score'].apply(lambda score: 'Positive' if score > 3 else 'Negative' if score < 3 else 'Neutral')

# Save the preprocessed data
data.to_csv('AmazonReview\preprocessed_reviews.csv', index=False)
