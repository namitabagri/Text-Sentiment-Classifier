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
from sklearn.decomposition import TruncatedSVD

# [CORRECTION 1] Simplified NLTK resource download.
# This is a more robust way to ensure the packages are present.
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


# Load models and encoders
@st.cache_resource
def load_models():
    """Load all necessary models, vectorizers, and transformers from disk."""
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    target_encoder = joblib.load('target_encoder.pkl')

    # [CORRECTION 2] Load the FITTED SVD transformer from the notebook.
    # This is crucial for making correct predictions.
    svd = joblib.load('svd_transformer.pkl')

    return model, vectorizer, target_encoder, svd


model, vectorizer, target_encoder, svd = load_models()


# Text cleaning function (no changes needed here)
def clean(text):
    """Cleans and preprocesses a single text string."""
    lm = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
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
    """Predicts sentiment for a given text and title."""
    # Clean the text
    cleaned_text = clean(text)

    # Vectorize the text using the loaded vectorizer
    text_vec = vectorizer.transform([cleaned_text])

    # Encode the title using the loaded target encoder
    title_df = pd.DataFrame({'title': [title]})
    title_encoded = target_encoder.transform(title_df)

    # Combine features
    features = hstack([title_encoded, text_vec]).tocsr()

    # Normalize features
    normalizer = Normalizer(norm='l2')
    features_normalized = normalizer.transform(features)

    # [CORRECTION 3] Use the loaded SVD object to TRANSFORM the new data.
    # We must not use .fit() or .fit_transform() here. The SVD model is already trained.
    features_reduced = svd.transform(features_normalized)

    # Make prediction
    prediction = model.predict(features_reduced)
    probabilities = model.predict_proba(features_reduced)

    return prediction[0], probabilities


# --- Streamlit UI (no changes needed here) ---
st.set_page_config(page_title="Twitter Sentiment Classifier", layout="wide")
st.title("🐦 Twitter Sentiment Classifier")
st.write("This app predicts the sentiment of tweets about various products and companies.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    title_options = [
        'Borderlands', 'Facebook', 'Amazon', 'Microsoft', 'Google', 'CallOfDuty',
        'TomClancysRainbowSix', 'MaddenNFL', 'FIFA', 'AssassinsCreed', 'CS-GO'
    ]
    title = st.selectbox(
        "Select the topic/company of the tweet:",
        options=title_options
    )

with col2:
    user_input = st.text_area("Enter the tweet text to analyze:", "I am really loving the new update, it's so good!",
                              height=150)

if st.button("Analyze Sentiment", type="primary"):
    if user_input and title:
        with st.spinner('Analyzing...'):
            prediction, probabilities = predict_sentiment(user_input, title)

            st.subheader("Prediction Result")

            if prediction == 'Positive':
                st.success(f"**Predicted Sentiment: {prediction}** 👍")
            elif prediction == 'Negative':
                st.error(f"**Predicted Sentiment: {prediction}** 👎")
            elif prediction == 'Neutral':
                st.info(f"**Predicted Sentiment: {prediction}** 😐")
            else:  # Irrelevant
                st.warning(f"**Predicted Sentiment: {prediction}** 🤷")

            st.write("**Probability Distribution:**")
            prob_df = pd.DataFrame(probabilities, columns=model.classes_, index=["Probability"]).T
            prob_df = prob_df.sort_values(by="Probability", ascending=False)
            st.bar_chart(prob_df)
    else:
        st.warning("Please select a topic and enter some text to analyze.")

st.sidebar.header("About")
st.sidebar.info("""
This sentiment classifier was trained on the 'Twitter Entity Sentiment Analysis' dataset. The notebook demonstrates the use of a Logistic Regression model with the Bag-of-Words text representation technique.

**Key Steps:**
- **Text Cleaning:** Lowercasing, punctuation/stopword removal, and lemmatization.
- **Feature Engineering:**
    - `CountVectorizer` (Bag-of-Words) for tweet text.
    - `TargetEncoder` for the tweet's topic/title.
- **Modeling:** A `LogisticRegression` model is used for classification.
""")