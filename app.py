import os
import asyncio
import feedparser
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

# ---------------------------------------------------
# Ensure asyncio event loop exists for Streamlit
# ---------------------------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Disable parallelism warnings from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------
# Initialize Hugging Face Pipelines (CPU-only)
# ---------------------------------------------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # Force CPU usage
)

summarization_model = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # Force CPU usage
)

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
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
            "summary": entry.summary  # Use provided summary/description
        })
    return articles

def analyze_article(text):
    """
    Analyze sentiment and generate a summary for the provided text.
    Returns a tuple: (sentiment label, confidence score, summary text)
    """
    try:
        sentiment_result = sentiment_model(text)[0]
        sentiment_label = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
    except Exception as e:
        sentiment_label, sentiment_score = "Error", 0

    try:
        summary_result = summarization_model(text, max_length=50, min_length=10, do_sample=False)
        summary_text = summary_result[0]["summary_text"]
    except Exception as e:
        summary_text = "Error during summarization."

    return sentiment_label, sentiment_score, summary_text

def generate_wordcloud(articles):
    """
    Generate a word cloud from the summaries of articles.
    Returns a matplotlib figure.
    """
    combined_text = " ".join(article["summary"] for article in articles)
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# ---------------------------------------------------
# Streamlit App Interface
# ---------------------------------------------------
st.set_page_config(page_title="AI News Analyzer", layout="wide")
st.title("📰 AI-Powered News Analyzer")

# User input: Topic selection
topic = st.text_input("Enter a topic to analyze:", "technology")

if st.button("Fetch & Analyze"):
    st.info(f"Fetching news for topic: {topic}")
    articles = fetch_news(topic)

    if not articles:
        st.error("No articles found. Try a different topic.")
    else:
        # Process and display each article's analysis
        for article in articles:
            st.subheader(article["title"])
            sentiment, confidence, summary = analyze_article(article["summary"])
            st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
            st.write(f"**Summary:** {summary}")
            st.markdown("---")

        # Generate and display the word cloud from article summaries
        st.subheader("Word Cloud from Article Summaries")
        wc_fig = generate_wordcloud(articles)
        st.pyplot(wc_fig)

st.markdown("⚡ Powered by AI & Streamlit | Built by [Abubakr Farid]")
