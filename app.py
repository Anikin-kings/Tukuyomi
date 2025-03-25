import os
import asyncio
import feedparser
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch
from transformers import pipeline

# -----------------------------
# Fix for asyncio event loop issues in Streamlit
# -----------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -----------------------------
# Explicit Environment & Model Setup
# -----------------------------
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"  # For clearer error reporting

# Use GPU if available, otherwise CPU
device_flag = 0 if torch.cuda.is_available() else -1

# Initialize Hugging Face pipelines with explicit models and device
sentiment_pipeline_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device_flag
)
summarizer_pipeline_model = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device_flag
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
            "summary": entry.summary  # summary/description provided by the RSS feed
        })
    return articles

def analyze_text(text):
    """
    Analyze sentiment and generate a summary for the provided text.
    Returns a tuple: (sentiment label, sentiment confidence, summary text)
    """
    try:
        sentiment_result = sentiment_pipeline_model(text)[0]
        sentiment_label = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
    except Exception as e:
        sentiment_label, sentiment_score = "Error", 0

    try:
        summary_result = summarizer_pipeline_model(text, max_length=50, min_length=10, do_sample=False)
        summary_text = summary_result[0]["summary_text"]
    except Exception as e:
        summary_text = "Error during summarization."

    return sentiment_label, sentiment_score, summary_text

def create_wordcloud(articles):
    """
    Create a word cloud from the summaries of the articles.
    Returns a matplotlib figure.
    """
    combined_text = " ".join([article["summary"] for article in articles])
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# -----------------------------
# Streamlit App Interface
# -----------------------------
st.set_page_config(page_title="AI News Sentiment Analyzer", layout="wide")
st.title("📰 AI-Powered News Sentiment Analyzer")

# Topic input from user
topic = st.text_input("Enter a topic to analyze:", value="technology")

if st.button("Analyze News"):
    st.info(f"Fetching news articles for **{topic}**...")
    articles = fetch_news(topic)

    if not articles:
        st.error("No articles found. Try a different topic.")
    else:
        # Container for displaying article results
        for article in articles:
            st.subheader(article["title"])
            # Use the article's summary for sentiment and summarization analysis
            sentiment, confidence, summary = analyze_text(article["summary"])
            st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
            st.write(f"**Summary:** {summary}")
            st.markdown("---")

        # Display the word cloud
        st.subheader("Word Cloud from Article Summaries")
        wc_fig = create_wordcloud(articles)
        st.pyplot(wc_fig)

st.markdown("⚡ Powered by AI & Streamlit | Built by [Your Name]")
