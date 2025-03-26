import os
import asyncio
import feedparser
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# -----------------------------
# Ensure NLTK VADER Lexicon is downloaded
# -----------------------------
nltk.download('vader_lexicon')

# -----------------------------
# Fix for asyncio event loop issues in Streamlit
# -----------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -----------------------------
# Initialize VADER and Summarization Pipeline
# -----------------------------
sentiment_analyzer = SentimentIntensityAnalyzer()

summarization_model = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # Force CPU usage
)

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_news(topic="technology", limit=5):
    """
    Fetch news articles from Google News RSS for a given topic.
    Returns a list of dictionaries with 'title' and 'summary'.
    """
    rss_url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:limit]:
        articles.append({
            "title": entry.title,
            "summary": entry.summary  # Using the RSS-provided summary/description
        })
    return articles

def analyze_text(text):
    """
    Analyze sentiment using VADER and generate a summary using Hugging Face.
    Returns a tuple: (sentiment label, confidence score, summary text).
    """
    # VADER sentiment analysis
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    compound = sentiment_scores["compound"]
    if compound >= 0.05:
        sentiment_label = "POSITIVE"
    elif compound <= -0.05:
        sentiment_label = "NEGATIVE"
    else:
        sentiment_label = "NEUTRAL"
    confidence = abs(compound)
    
    # Generate summary using Hugging Face summarizer
    try:
        summary_result = summarization_model(text, max_length=50, min_length=10, do_sample=False)
        summary_text = summary_result[0]["summary_text"]
    except Exception as e:
        summary_text = "Error during summarization."
    
    return sentiment_label, confidence, summary_text

def generate_wordcloud(articles):
    """
    Generate a word cloud from the summaries of the articles.
    Returns a matplotlib figure.
    """
    combined_text = " ".join(article["summary"] for article in articles)
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# -----------------------------
# Streamlit App Interface
# -----------------------------
st.set_page_config(page_title="AI News Analyzer with VADER", layout="wide")
st.title("📰 AI-Powered News Analyzer with VADER")

# User input: Topic selection
topic = st.text_input("Enter a topic to analyze:", "technology")

if st.button("Fetch & Analyze"):
    st.info(f"Fetching news for topic: {topic}")
    articles = fetch_news(topic)
    
    if not articles:
        st.error("No articles found. Try a different topic.")
    else:
        # Process and display each article
        for article in articles:
            st.subheader(article["title"])
            sentiment, confidence, summary = analyze_text(article["summary"])
            st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
            st.write(f"**Summary:** {summary}")
            st.markdown("---")
        
        # Display word cloud from article summaries
        st.subheader("Word Cloud from Article Summaries")
        wc_fig = generate_wordcloud(articles)
        st.pyplot(wc_fig)

st.markdown("⚡ Powered by AI & Streamlit | Built by [Abubakr Farid]")
