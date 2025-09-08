import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from category_encoders import TargetEncoder
from scipy.sparse import hstack
from sklearn.preprocessing import Normalizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Load models
@st.cache_resource
def load_models():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    return model, vectorizer, target_encoder


model, vectorizer, target_encoder = load_models()


# Text cleaning function (same as in notebook)
def clean(text):
    stop_words = set(stopwords.words('english'))
    lm = WordNetLemmatizer()
    text = text.lower()
    no_punct = re.sub(r'[^a-z\\s]', '', text)
    words = word_tokenize(no_punct)
    words = [i for i in words if i not in stop_words]
    pos_tags = nltk.pos_tag(words)

    words = [
        lm.lemmatize(word, pos='v') if tag.startswith('V') else
        lm.lemmatize(word, pos='n') if tag.startswith('N') else
        lm.lemmatize(word, pos='a') if tag.startswith('R') else
        lm.lemmatize(word)
        for word, tag in pos_tags
    ]

    clean_words = ' '.join(words)
    return clean_words


# Prediction function
def predict_sentiment(text, title):
    # Clean the text
    cleaned_text = clean(text)

    # Vectorize the text
    text_vec = vectorizer.transform([cleaned_text])

    # Encode the title
    title_encoded = target_encoder.transform(pd.DataFrame([title]))
    #title_encoded = target_encoder.transform(pd.DataFrame({'title': [title]}))

    # Combine features
    features = hstack([title_encoded, text_vec])

    # Normalize
    normalizer = Normalizer(norm='l2')
    features_normalized = normalizer.transform(features)

    # Make prediction
    prediction = model.predict(features_normalized)
    probabilities = model.predict_proba(features_normalized)

    return prediction[0], probabilities


# Streamlit UI
st.title("Twitter Sentiment Classifier")
st.write("This app predicts the sentiment of tweets about products/companies.")

# Input fields
title = st.selectbox(
    "Select the product/company this tweet is about:",
    ['Borderlands', 'Facebook', 'Amazon', 'Microsoft']  # Add more from your data
)

user_input = st.text_area("Enter the tweet text to analyze:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Make prediction
        prediction, probabilities = predict_sentiment(user_input, title)

        # Display results
        st.subheader("Prediction Result")

        sentiment_labels = ['Negative', 'Neutral', 'Positive', 'Irrelevant']
        prob_dict = {label: prob for label, prob in zip(sentiment_labels, probabilities[0])}

        st.write(f"**Predicted Sentiment:** {prediction}")

        st.write("**Probability Distribution:**")
        for label, prob in prob_dict.items():
            st.write(f"{label}: {prob:.2%}")

        # Visualize probabilities
        st.bar_chart(prob_dict)
    else:
        st.warning("Please enter some text to analyze.")

# Add some info about the model
st.sidebar.header("About")
st.sidebar.write("""
This sentiment classifier was trained on Twitter data using:
- Logistic Regression
- Count Vectorizer for text features
- Target Encoding for product names
""")