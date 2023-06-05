<div align="center">

# Sentiment Analysis Project Documentation
# Introduction of NLP and its Use Cases

</div>

```Natural Language Processing (NLP)``` is a branch of artificial intelligence (AI) that focuses on the interaction between computers and human language. It involves the analysis, understanding, and generation of natural language text or speech. NLP combines techniques from various fields, including linguistics, computer science, and statistics, to enable computers to understand and process human language.

### What is NLP?

NLP aims to bridge the gap between human language and computer understanding. It involves several tasks, such as text classification, named entity recognition, sentiment analysis, machine translation, question answering, chatbots, and more. NLP systems analyze and interpret human language data in a way that allows computers to perform tasks traditionally done by humans.

### Training NLP Models

NLP models are trained using large amounts of labeled data. This data is used to teach the models to recognize patterns, relationships, and semantic meaning within text. The training process typically involves preprocessing the data, extracting relevant features, and training a machine learning or deep learning model. The models learn from the labeled data to make predictions or perform specific tasks.

### Use Cases and Applications of NLP

NLP has numerous applications across various industries and domains. Some of the key use cases of NLP include:

1. ```Sentiment Analysis```: Analyzing and determining the sentiment or emotional tone of a given text, such as customer reviews or social media posts.

2. ```Machine Translation```: Translating text or speech from one language to another, enabling communication across language barriers.

3. ```Named Entity Recognition (NER)```: Identifying and classifying named entities (such as names, locations, organizations) within a text.

4. ```Text Summarization```: Generating concise summaries or abstracts of longer text documents.

5. ```Question Answering```: Building systems that can understand questions in natural language and provide accurate answers.

6. ```Chatbots and Virtual Assistants```: Developing conversational agents that can understand and respond to user queries or perform tasks.

7. ```Information Extraction```: Extracting structured information from unstructured text, such as extracting entities, relationships, or facts.

### Organizations Using NLP

NLP techniques are widely used by organizations across different industries. Some prominent examples include:

- ```Google```: Google uses NLP for various applications, such as search algorithms, language translation, voice assistants (Google Assistant), and sentiment analysis.

- ```Amazon```: Amazon employs NLP for customer reviews analysis, product recommendations, voice assistants (Alexa), and understanding customer queries.

- ```Facebook```: Facebook uses NLP for content moderation, sentiment analysis, chatbots, language translation, and personalized content recommendations.

- ```Microsoft```: Microsoft utilizes NLP in products like Microsoft Office, Bing search engine, language translation, voice assistants (Cortana), and sentiment analysis.

- ```Apple```: Apple integrates NLP in Siri, its voice assistant, for natural language understanding, question answering, and completing tasks based on user commands.

### Essential Components in NLP Projects

NLP projects typically involve several essential components, including:

- ```Text Preprocessing```: Cleaning and transforming raw text data by removing noise, normalizing text, handling punctuation, converting to lowercase, tokenization, and removing stop words.

- ```Feature Extraction```: Converting preprocessed text into numerical features that machine learning models can process, such as bag-of-words, TF-IDF, word embeddings, or contextual embeddings.

- ```Machine Learning Models```: Training and evaluating various models such as logistic regression, Naive Bayes, support vector machines (SVM), recurrent neural networks (RNNs), or transformer-based models like BERT.

- ```Evaluation Metrics```: Assessing the performance of NLP models using metrics like accuracy, precision, recall, F1-score, or area under the receiver operating characteristic curve (AUC-ROC).

- ```Deployment```: Integrating the trained NLP model into production systems, building APIs, or developing applications to make predictions on new, unseen data.

In this project, we will explore sentiment analysis, one of the essential use cases of NLP, and leverage NLP techniques to classify the sentiment of customer reviews using the "Amazon Fine Food Reviews" dataset.

---
<div align="center">

# 1. Project Overview

</div>

The sentiment analysis project aims to analyze the sentiment of customer reviews using the ["Amazon Fine Food Reviews" ](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle. Sentiment analysis, also known as opinion mining, is a technique used to extract subjective information from text and determine the sentiment expressed within it.

## Problem Statement
The problem we are addressing in this project is to predict the sentiment (positive, negative, or neutral) of customer reviews based on their textual content. By analyzing the sentiment of the reviews, we can gain insights into customer opinions and attitudes towards the food products available on Amazon. This information can be valuable for businesses in understanding customer satisfaction, identifying areas for improvement, and making data-driven decisions to enhance their products and services.

## Dataset Description
The ["Amazon Fine Food Reviews" ](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset is a collection of reviews of various food products available on Amazon. It consists of thousands of reviews along with their corresponding ratings. The dataset provides valuable insights into customer opinions and preferences.

## Purpose
The primary objective of this project is to build a sentiment analysis model that can accurately classify the sentiment of customer reviews as positive, negative, or neutral. By understanding the sentiment expressed in the reviews, businesses can gain valuable insights into customer satisfaction, product improvements, and marketing strategies.

## Relevance
Sentiment analysis has become increasingly important in today's digital era, where customer feedback and online reviews greatly influence purchasing decisions. By automating the sentiment analysis process, businesses can save time and resources while gaining valuable insights into customer sentiments at scale.

## Key Deliverables
The main deliverables of this sentiment analysis project include:
- Preprocessing and cleaning of the ["Amazon Fine Food Reviews" ](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset
- Building and training sentiment analysis models
- Evaluation of model performance using appropriate metrics
- Interpretation and analysis of the results
- Documentation of the project process, findings, and insights

By completing these deliverables, we aim to provide an efficient and accurate sentiment analysis solution that can be applied to other text datasets and contribute to the understanding of customer sentiments.

---

<div align="center">

# 2. Installation and Set-up

</div>

## Prerequisites
Before getting started with the installation and setup process, ensure that you have the following prerequisites:

- **Python**: Make sure you have Python installed on your system. You can download and install Python from the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)

## Installation Steps
Follow the steps below to install and set up the required dependencies for the sentiment analysis project:

1. **Clone the repository**: Start by cloning the project repository from GitHub by running the following command in your terminal:

```git clone https://github.com/divyanv/SentimentAnalysis.git```

This will create a local copy of the project on your machine.

2. **Create a virtual environment**: It is recommended to create a virtual environment to isolate the project dependencies. Navigate to the project directory and run the following command to create a virtual environment:

```pip install virtualenv ```

```python -m venv venv ```

3. **Activate the virtual environment**: Activate the virtual environment by running the appropriate command based on your operating system:
- On Windows:

```./venv/scripts/activate  ```

- On macOS and Linux:
  
```source env/bin/activate```

4. **Install the required packages**: Once the virtual environment is activated, install the necessary packages by running the following command:

```pip install -r requirements.txt```

This command will install all the dependencies listed in the `requirements.txt` file.

5. **Download the dataset**: Download the ["Amazon Fine Food Reviews"](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle . Extract the dataset and place it in a directory within the project.

6. **Run the project**: You are now ready to run the sentiment analysis project! Execute the main script or notebook file to perform sentiment analysis on the ["Amazon Fine Food Reviews" ](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset.

## Configuration

The following configurations can be modified in the project:

1. **Dataset Configuration**: Place the downloaded dataset in a directory within the project. Ensure that the file names and formats are consistent with the code's expectations.

2. **Model Configuration**: The LSTM model with Word2Vec embeddings can be customized by adjusting the following parameters in the code:
   - Embedding size
   - LSTM layer size
   - Optimizer

3. **Training Configuration**: Modify the training-related configurations such as:
   - Batch size
   - Number of epochs
   - Early stopping criteria

4. **Preprocessing Configuration**: Customize the text preprocessing steps, such as using a different stop word list or adjusting the threshold for rare word removal, by modifying the  `preprocess_text()` and `process_text()` helper functions accordingly.

5. **Hardware Configuration**: The code supports GPU acceleration if available. Ensure that the necessary GPU drivers are installed to take advantage of GPU training. 

(Alternatively, you can utilize cloud-based GPU instances for faster training.)

6. **Output and Visualization Configuration**: The project generates various outputs and visualizations, such as training accuracy and loss plots, and a confusion matrix. Customize the plot sizes, color maps, and visualization styles in the code.


---
<div align="center">

# 3. Data Description

</div>

## Data Overview

The "Amazon Fine Food Reviews" dataset contains a vast collection of customer reviews for food products sold on Amazon. It includes over 500,000 reviews, spanning a period of more than 10 years from October 1999 to October 2012. Each review is accompanied by various attributes, including the review text, reviewer's ID, product ID, and timestamps.

- Number of reviews: 568,454
- Number of users: 256,059
- Number of products: 74,258
- Timespan: Oct 1999 - Oct 2012
- Number of Attributes/Columns in data: 10

## Attribute Information

1. **Id**
2. **ProductId** - unique identifier for the product
3. **UserId** - unique identifier for the user
4. **ProfileName**
5. **HelpfulnessNumerator** - number of users who found the review helpful
6. **HelpfulnessDenominator** - number of users who indicated whether they found the review helpful or not
7. **Score** - rating between 1 and 5
8. **Time** - timestamp for the review
9. **Summary** - brief summary of the review
10. **Text** - text of the review

---
<div align="center">

# 4. Data Source, Format, Size, and Preprocessing Steps Applied

</div>


The dataset was obtained from Kaggle and is provided in a compressed format (e.g., .zip). After extraction, the dataset consists of a CSV file containing the review data. The file is in tabular format, with each row representing a review and its corresponding attributes.

Before utilizing the dataset, several preprocessing steps were applied, including:

<!-- # NLP Preprocessing Steps

To prepare the text data for sentiment analysis, the following NLP preprocessing steps are typically applied in order:

1. **Tokenization**: Tokenization is the process of splitting the text into individual words or tokens. This step enables further analysis at the word level and serves as the foundation for subsequent preprocessing steps.

2. **Text Cleaning**: Text cleaning involves removing any special characters, punctuation marks, and numeric digits from the text. It helps standardize the text and remove unnecessary noise.

3. **Lowercasing**: Converting all the text to lowercase helps in achieving case insensitivity and avoiding duplicate representations of words due to different capitalizations.

4. **Stop Word Removal**: Stop words are common words that do not carry significant meaning in the context of sentiment analysis, such as "and," "the," or "is." Removing these words reduces noise and improves computational efficiency during analysis.

5. **Normalization**: Normalization aims to reduce word variations by converting words to their base or root form. Techniques such as stemming and lemmatization are commonly used for this purpose.

6. **Removing Non-Alphanumeric Tokens**: Removing non-alphanumeric tokens, such as URLs, special symbols, or emojis, helps eliminate irrelevant information and focuses the analysis on meaningful text.

7. **Handling Abbreviations and Contractions**: Abbreviations and contractions are expanded to their full forms to ensure consistent representation of words and improve the accuracy of sentiment analysis.

8. **Removing Custom Stop Words**: In addition to standard stop words, domain-specific or task-specific stop words can be identified and removed to further refine the text data.

9. **Handling Negation**: Negation words, such as "not" or "no," can significantly affect the sentiment expressed in a sentence. Identifying and appropriately handling negation helps capture their impact on sentiment analysis.

10. **Feature Engineering**: Additional features can be extracted from the text data, such as n-grams, part-of-speech tags, or sentiment scores of individual words. These features can provide additional context and improve the performance of sentiment analysis models.

By applying these preprocessing steps in order, the text data is transformed into a clean and normalized format suitable for sentiment analysis tasks.

-->

## NLP Preprocessing Steps

To prepare the text data for analysis, the following preprocessing steps are commonly applied in Natural Language Processing (NLP):

1. **Text Cleaning and Normalization**:
   - Convert the text to lowercase.
   - Remove special characters, such as punctuation marks and symbols.
   - Remove numeric digits or replace them with placeholders if they are not relevant to the analysis.
   - Handle contractions, such as converting "can't" to "cannot."

2. **Tokenization**:
   - Split the text into individual words or tokens.
   - Consider using advanced tokenization techniques, such as subword tokenization, for languages with complex word structures.

3. **Stop Word Removal**:
   - Remove commonly used words with little semantic value, known as stop words (e.g., "and," "the," "is").
   - Utilize established stop word lists or create custom stop word lists tailored to the specific domain or analysis.

4. **Part-of-Speech (POS) Tagging**:
   - Assign grammatical tags to each word in the text, such as noun, verb, adjective, or adverb.
   - POS tagging is helpful for understanding the syntactic structure of the text and can be used for further analysis, such as identifying noun phrases or verb phrases.

5. **Lemmatization or Stemming**:
   - Reduce words to their base or root form to normalize variations.
   - Lemmatization considers the morphological analysis of words to determine their base form (e.g., "running" to "run").
   - Stemming applies heuristic rules to remove prefixes or suffixes from words (e.g., "running" to "run").

6. **Word Sense Disambiguation**:
   - Resolve ambiguous words with multiple meanings based on the context.
   - Techniques like word embeddings or lexical databases can be used to disambiguate word senses.

7. **Entity Recognition**:
   - Identify and classify named entities, such as person names, locations, organizations, or dates, within the text.
   - Named entity recognition helps extract structured information from unstructured text data.

8. **Feature Engineering**:
   - Generate additional features or representations from the text, such as n-grams, bag-of-words, or TF-IDF (Term Frequency-Inverse Document Frequency).
   - Feature engineering enriches the text data representation and captures important information for downstream analysis.

These preprocessing steps are performed to clean, standardize, and transform the text data into a suitable format for NLP analysis and modeling.


---
<div align="center">

# 5. Methodology and Models

</div>


Sentiment analysis in natural language processing (NLP) involves the use of various methodologies and models to determine the sentiment or opinion expressed in a piece of text. The Amazon Fine Dine dataset is a popular dataset used for sentiment analysis tasks, specifically focusing on customer reviews of fine dining restaurants on Amazon. In this context, sentiment analysis aims to classify the sentiment of the reviews as positive, negative, or neutral.

## Methodology

1. **Data Preprocessing**: The first step in sentiment analysis involves data preprocessing, which includes cleaning and transforming the raw text data into a suitable format for analysis. This typically involves removing noise, such as punctuation, special characters, and HTML tags, as well as handling capitalization, tokenization, and stemming/lemmatization to reduce words to their base forms.

2. **Feature Extraction**: Once the text data has been preprocessed, the next step is to extract relevant features from the text that can be used to train machine learning models. Common techniques for feature extraction in NLP include bag-of-words, TF-IDF (Term Frequency-Inverse Document Frequency), and word embeddings such as Word2Vec or GloVe. These techniques transform the text into numerical representations that capture semantic information.

3. **Model Selection**: After feature extraction, a suitable machine learning model is chosen to train on the extracted features. Various models have been employed for sentiment analysis, including:

   - Naive Bayes: A probabilistic model based on Bayes' theorem, which assumes independence between features. It is simple and efficient, making it a popular choice for sentiment analysis tasks.
   
   - Support Vector Machines (SVM): A binary classification model that finds an optimal hyperplane to separate different classes. SVMs can effectively handle high-dimensional feature spaces and are known for their good generalization performance.
   
   - Recurrent Neural Networks (RNN): A type of neural network that can capture sequential information in the input data. Models like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are often used for sentiment analysis due to their ability to handle dependencies between words in a sentence.
   
   - Convolutional Neural Networks (CNN): Typically used for image processing, CNNs can also be applied to text analysis. They utilize convolutional layers to extract features from n-grams of words in the input text, making them effective for sentiment analysis tasks.
   
   - Transformer-based Models: Transformers, such as the popular BERT (Bidirectional Encoder Representations from Transformers), have revolutionized NLP tasks. These models capture contextual information by considering the entire input sequence, resulting in state-of-the-art performance in sentiment analysis and other NLP tasks.

4. **Model Training and Evaluation**: The selected model is trained on the preprocessed data with the extracted features. The dataset is usually split into training, validation, and test sets. The training set is used to optimize the model's parameters through techniques like gradient descent, while the validation set is used to fine-tune hyperparameters and prevent overfitting. Finally, the model's performance is evaluated on the test set using metrics such as accuracy, precision, recall, and F1 score.

5. Model Deployment and Application: Once the model has been trained and evaluated, it can be deployed to perform sentiment analysis on new, unseen data. This could involve predicting sentiment labels for individual reviews or aggregating sentiment scores for broader analysis, such as sentiment trends over time or sentiment analysis of large-scale datasets.

## Models for Amazon Fine Dine Sentiment Analysis

The choice of specific models for sentiment analysis on the Amazon Fine Dine dataset would depend on various factors, including the size of the dataset, the computational resources available, and the desired level of performance. Depending on these considerations, a combination of traditional machine learning models (such as Naive Bayes or SVM) and deep learning models (such as RNNs, CNNs, or Transformer-based models like BERT) can be employed.

The performance of the models can be assessed through rigorous evaluation techniques, including cross-validation and benchmarking against existing state-of-the-art sentiment analysis models. It is important to continuously iterate and fine-tune the models based on feedback and evaluation results to achieve the best possible sentiment analysis performance on the Amazon Fine Dine dataset.


---
<div align="center">

# 6. Feature Extraction 

</div>

## Feature Extraction in NLP Sentiment Analysis from Amazon Fine Dine Dataset

Feature extraction is a crucial step in natural language processing (NLP) sentiment analysis tasks, including the analysis of sentiment in Amazon Fine Dine dataset reviews. The goal of feature extraction is to transform raw textual data into a numerical representation that machine learning algorithms can effectively process and analyze.

In the context of NLP sentiment analysis, feature extraction involves identifying and extracting relevant information or features from text that can be indicative of sentiment. These features provide the necessary input for machine learning models to learn patterns and make predictions about the sentiment expressed in the text.

## Common Feature Extraction Techniques

1. **Bag-of-Words (BoW):** The BoW model represents text as a collection of unique words, disregarding grammar and word order. Each review is transformed into a vector, where each element represents the frequency or presence of a word in the review. Stop words (common words like "the," "and," etc.) are often removed to reduce noise.

2. **TF-IDF (Term Frequency-Inverse Document Frequency):** TF-IDF is similar to BoW, but it also takes into account the importance of a word in a particular review and the entire dataset. It assigns higher weights to words that are more frequent in a specific review but less frequent in the entire dataset. This helps in capturing the relative importance of words.

3. **Word Embeddings:** Word embeddings, such as Word2Vec or GloVe, represent words as dense numerical vectors in a continuous vector space. These vectors are trained on large corpora and capture semantic relationships between words. Sentiment analysis models can use pre-trained word embeddings or train their own embeddings specific to the Amazon Fine Dine dataset.

4. **Part-of-Speech (POS) Tags:** POS tagging involves labeling each word in a text with its corresponding part of speech (e.g., noun, verb, adjective). POS tags can be used as features to capture grammatical patterns or the role of specific words in expressing sentiment.

5. **N-grams:** N-grams are contiguous sequences of N words in a text. By considering not just individual words but also combinations of words, N-grams can capture contextual information and dependencies between words. For sentiment analysis, commonly used N-gram models include unigrams (single words), bigrams (two-word sequences), and trigrams (three-word sequences).

6. **Sentiment Lexicons:** Sentiment lexicons are curated dictionaries or lists of words with assigned sentiment scores. These lexicons contain words annotated with positive or negative sentiment polarity. By matching words from the text with entries in the sentiment lexicon, sentiment scores can be assigned to the text and used as features for sentiment analysis.

These feature extraction techniques provide a way to represent textual data in a format suitable for machine learning algorithms. The extracted features can then be used as input to train models such as logistic regression, support vector machines, or neural networks to predict sentiment labels (positive, negative, or neutral) for the Amazon Fine Dine dataset reviews.


---
<div align="center">

# 7. Model Traning 

</div>

# Model Training for NLP Sentiment Analysis on Amazon Fine Dine Dataset

Sentiment analysis is a popular Natural Language Processing (NLP) task that involves determining the sentiment expressed in a given text. One common approach to sentiment analysis is to train a machine learning model using a labeled dataset, such as the Amazon Fine Dine dataset. In this article, we will focus on the model training process for NLP sentiment analysis specifically using this dataset.

## Dataset Preparation

Before diving into model training, it is crucial to prepare the dataset appropriately. The Amazon Fine Dine dataset consists of customer reviews labeled with sentiment categories like positive, negative, and neutral. The dataset is typically split into three subsets: training, validation, and testing.

### 1. Training-Test-Validation Split

To evaluate the performance of the trained model accurately, it is essential to divide the dataset into these three subsets. The training set is used to train the model, the validation set is used for hyperparameter tuning and model selection, and the test set is used for final evaluation.

The commonly used split ratio is 70% for training, 15% for validation, and 15% for testing. This split ensures that the model is trained on a sufficient amount of data while providing enough samples for validation and testing.

### 2. Text Preprocessing

Once the dataset is split, text preprocessing techniques are applied to clean and normalize the text data. This may involve steps like removing special characters, converting text to lowercase, tokenization (splitting text into individual words or tokens), removing stopwords (commonly occurring words with little significance), and performing stemming or lemmatization to reduce words to their base form.

## Model Training

After preparing the dataset, the model training process begins. The specific algorithm chosen for sentiment analysis depends on the requirements and characteristics of the task. Commonly used algorithms for sentiment analysis include Support Vector Machines (SVM), Naive Bayes, and Recurrent Neural Networks (RNN) such as Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRU).

### 1. Feature Extraction

To represent textual data in a format suitable for machine learning algorithms, feature extraction techniques are employed. The Bag-of-Words (BoW) model or more advanced methods like Term Frequency-Inverse Document Frequency (TF-IDF) can be used to convert text into numerical feature vectors.

### 2. Model Optimization

Optimization techniques are applied to fine-tune the model's performance. Hyperparameters, such as learning rate, regularization strength, and model architecture, are adjusted to find the optimal configuration. This process is often carried out using the validation set.

Grid search or randomized search can be used to systematically explore different combinations of hyperparameters and select the ones that yield the best performance. Additionally, techniques like cross-validation can be used to assess the model's performance on different subsets of the training data.

### 3. Training and Evaluation

The training phase involves feeding the preprocessed dataset into the model. The model learns to associate the features extracted from the text with the sentiment labels. The loss function, such as binary cross-entropy or softmax loss, is used to measure the difference between predicted and actual labels. The model's parameters are updated through techniques like gradient descent to minimize the loss.

During training, the model's performance is monitored on the validation set to prevent overfitting. Overfitting occurs when the model becomes too specialized in the training data and fails to generalize well to unseen data. Regularization techniques like dropout or early stopping can be employed to mitigate overfitting.

After the training is complete, the final evaluation is carried out using the test set. The accuracy, precision, recall, F1-score, or other appropriate metrics are calculated to measure the model's performance on unseen data.

In conclusion, training an NLP sentiment analysis model using the Amazon Fine Dine dataset involves proper dataset splitting, text preprocessing, feature extraction, optimization of the model's hyperparameters, and evaluation on separate validation and test sets. The process aims to create a robust and accurate sentiment analysis model capable of predicting sentiment in text data from the Amazon Fine Dine dataset or similar sources.


---
<div align="center">

# 8. Result and Conclusion

</div>

**Results:**
- The dataset was successfully loaded and preprocessed, including tokenization, lowercasing, and removal of special characters and numbers.
- Sentiment scores were calculated for each text using the VADER sentiment analyzer, and the scores were mapped to positive, negative, and neutral sentiments based on a threshold.
- The text sequences were converted to numerical sequences and padded to ensure uniform length.
- The LSTM model with Word2Vec embeddings was trained on the dataset, achieving a certain accuracy and loss on the training and validation sets.
- The evaluation metrics, including accuracy, precision, recall, and F1-score, were calculated on the test set, and a confusion matrix was generated to visualize the model's performance.

**Conclusion:**
- The sentiment analysis task successfully classified the sentiment of the given texts into positive, negative, and neutral categories.
- The LSTM model with Word2Vec embeddings demonstrated its effectiveness in capturing the semantic meaning of words and achieving reasonable accuracy in sentiment prediction.
- The evaluation metrics and confusion matrix provide insights into the model's performance and can guide further improvements or applications of sentiment analysis in real-world scenarios.





