import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("model/fake_news_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detector")
st.write("Detect whether a news article is likely to be Fake or Real using NLP and Machine Learning.")

# News input
news_text = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste the news article here..."
)

# Prediction
if st.button("🔍 Predict News"):

    if not news_text.strip():
        st.warning("⚠️ Please enter a news article first.")

    else:
        # Convert text to TF-IDF
        news_tfidf = tfidf.transform([news_text])

        # Prediction
        prediction = model.predict(news_tfidf)[0]

        # Probability
        probabilities = model.predict_proba(news_tfidf)[0]
        confidence = max(probabilities) * 100

        st.divider()

        if prediction == 0:
            st.error("🔴 Prediction: FAKE NEWS")
        else:
            st.success("🟢 Prediction: REAL NEWS")

        st.info(f"Model confidence: **{confidence:.2f}%**")

        st.caption(
            "Note: This prediction is based on patterns learned from the training dataset "
            "and should not be treated as a factual verification of the article."
        )

# Information section
st.divider()

st.subheader("⚙️ How it works")

st.write("""
1. News text is cleaned and processed.
2. TF-IDF converts the text into numerical features.
3. A Random Forest classifier analyzes the features.
4. The system predicts whether the news is Fake or Real.
""")