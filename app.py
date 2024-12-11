from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Define the root route
@app.route('/')
def index():
    return 'Welcome to the sentiment analysis app!'

@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Text data is missing"}), 400
    
    text = data["text"]
    
    # Vectorize the text data
    text_vect = vectorizer.transform([text])
    
    # Predict sentiment
    sentiment = model.predict(text_vect)[0]
    
    # Calculate percentage of positive and negative sentiment
    probas = model.predict_proba(text_vect)[0]
    positive_percent = probas[model.classes_ == 'positive'][0] * 100
    negative_percent = probas[model.classes_ == 'negative'][0] * 100
    
    # Find the words with highest probability for each class
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_positive_words = feature_names[np.argsort(model.feature_log_prob_[1])[::-1][:5]]
    top_negative_words = feature_names[np.argsort(model.feature_log_prob_[0])[::-1][:5]]
    if positive_percent > 60:
        sentiment = "positive"
    elif negative_percent > 60:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return jsonify({
        
        "sentiment": sentiment,
        "positive_percent": positive_percent,
        "negative_percent": negative_percent,
        "positive_words": top_positive_words.tolist(),
        "negative_words": top_negative_words.tolist(),
        "reason": "The text is classified based on the probability of words occurring in the text and their respective contributions to the sentiment classification."
    })

if __name__ == "__main__":
    app.run(debug=True)
