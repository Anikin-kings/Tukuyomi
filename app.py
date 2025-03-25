import feedparser
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def fetch_google_news_rss(topic="technology", limit=5):
    """
    Fetch news articles using Google News RSS for the given topic.
    """
    rss_url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    entries = feed.entries[:limit]
    return entries

def analyze_article(article, sentiment_pipeline, summarizer):
    """
    Analyze the sentiment and generate a summary for a news article.
    Uses the article's summary for analysis.
    """
    headline = article.get("title", "No Title")
    # Use the article's summary (description) for NLP analysis.
    description = article.get("summary", "")
    
    # Limit input size if necessary
    snippet = description[:512]
    
    # Perform sentiment analysis
    sentiment_result = sentiment_pipeline(snippet)
    
    # Generate summary (if description is short, the summarizer may return a shorter version)
    summary_result = summarizer(description, max_length=50, min_length=25, do_sample=False)
    
    sentiment = sentiment_result[0]["label"]
    summary = summary_result[0]["summary_text"]
    
    return headline, sentiment, summary

def create_wordcloud(text):
    """
    Create and display a word cloud image from the provided text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of News Headlines")
    plt.show()

def main():
    topic = "technology"  # Change this to any topic you like
    articles = fetch_google_news_rss(topic, limit=5)

    if not articles:
        print("No articles found.")
        return

    # Initialize Hugging Face pipelines for sentiment analysis and summarization
    sentiment_pipeline_model = pipeline("sentiment-analysis")
    summarizer_model = pipeline("summarization")

    combined_headlines = ""
    
    print(f"\nNews Sentiment Analysis for '{topic}'\n" + "="*60)
    for article in articles:
        headline, sentiment, summary = analyze_article(article, sentiment_pipeline_model, summarizer_model)
        combined_headlines += " " + headline
        print(f"Headline: {headline}")
        print(f"Sentiment: {sentiment}")
        print(f"Summary: {summary}")
        print("-"*60)
    
    # Generate a word cloud from the combined headlines
    create_wordcloud(combined_headlines)

if __name__ == "__main__":
    main()
