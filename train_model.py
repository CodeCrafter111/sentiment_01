# train_model.py

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Step 2: Load and preprocess the Sentiment140 dataset
# Load the Sentiment140 dataset into a DataFrame
file_path = 'D:/sentiment_analysis_new/Twitter_sentiment_analysis_project_v2/Twitter_sentiment_analysis_project_v2/backend/training.1600000.processed.noemoticon.csv'

# Read the CSV file
data = pd.read_csv(file_path, encoding='latin-1', header=None)

# Rename columns
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Drop unnecessary columns
data = data[['sentiment', 'text']]

# Replace sentiment labels
data['sentiment'] = data['sentiment'].replace({0: 'negative', 4: 'positive'})

# Step 3: Split data into training and testing sets
X = data['text']                # Text data
y = data['sentiment']           # Sentiment labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize the text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Step 5: Train a machine learning model (Naive Bayes in this example)
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 7: Save the model for future use
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
