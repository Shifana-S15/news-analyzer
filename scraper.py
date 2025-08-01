import feedparser
from newspaper import Article
import pandas as pd
import os
from datetime import datetime

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# RSS feed sources
RSS_FEEDS = {
    'BBC': 'http://feeds.bbci.co.uk/news/world/rss.xml',
    'NDTV': 'https://feeds.feedburner.com/ndtvnews-top-stories',
    'The Hindu': 'https://www.thehindu.com/news/national/feeder/default.rss'
}

def scrape_article(url, source, fallback_entry=None):
    try:
        print(f"🟡 Trying: {url}")
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        return {
            'title': article.title,
            'text': article.text,
            'published': article.publish_date if article.publish_date else datetime.now(),
            'summary': article.summary,
            'keywords': article.keywords,
            'authors': article.authors,
            'url': url,
            'source': source
        }

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        if fallback_entry:
            print("🔁 Falling back to RSS summary...")
            return {
                'title': fallback_entry.get('title', 'No title'),
                'text': fallback_entry.get('summary', 'No text'),
                'published': fallback_entry.get('published', datetime.now()),
                'summary': fallback_entry.get('summary', 'No summary'),
                'keywords': [],
                'authors': [],
                'url': fallback_entry.get('link', url),
                'source': source
            }
        return None

def scrape_rss():
    articles = []

    for source, url in RSS_FEEDS.items():
        print(f"\n📡 Scraping source: {source}")
        feed = feedparser.parse(url)

        for entry in feed.entries[:10]:  # limit to 10 articles per source
            article_data = scrape_article(entry.link, source, fallback_entry=entry)
            if article_data:
                articles.append(article_data)

    return articles

def save_to_csv(articles, filename='data/articles.csv'):
    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False)
    print(f"\n✅ Articles saved to {filename}")

if __name__ == "__main__":
    articles = scrape_rss()

    if articles:
        save_to_csv(articles)
    else:
        print("⚠️ No articles scraped.")
