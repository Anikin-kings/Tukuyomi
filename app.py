import feedparser
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure the required packages are installed: feedparser, transformers, wordcloud, matplotlib
def fetch_google_news_rss(topic="Anime", limit=10):
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
    description = article.get("summary", "")

    # Limit input size if necessary
    snippet = description[:512]

    try:
        # Perform sentiment analysis
        sentiment_result = sentiment_pipeline(snippet)

        # Generate summary
        summary_result = summarizer(description, max_length=50, min_length=25, do_sample=False)

        sentiment = sentiment_result[0]["label"]
        summary = summary_result[0]["summary_text"]
    except PipelineException as e:
        sentiment = "Error"
        summary = "Error generating summary"
        print(f"Error during analysis: {e}")

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
    topic = "Anime"  # Change this to any topic you like
    articles = fetch_google_news_rss(topic, limit=10)

    if not articles:
        print("No articles found.")
        return

    try:
        # Initialize Hugging Face pipelines for sentiment analysis and summarization
        sentiment_pipeline_model = pipeline("sentiment-analysis")
        summarizer_model = pipeline("summarization")
    except Exception as e:
        print(f"Error initializing Hugging Face pipelines: {e}")
        return

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
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
