import pandas as pd
from textblob import TextBlob
import spacy
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Download required resources
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Load the scraped articles
def load_articles(path='data/articles.csv'):
    return pd.read_csv(path)

# Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Named Entity Recognition
def extract_entities(text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]))

# Keyword Extraction
def extract_top_keywords(text, top_n=10):
    words = text.lower().split()
    filtered = [word for word in words if word.isalpha() and word not in stop_words]
    freq = Counter(filtered)
    return [word for word, count in freq.most_common(top_n)]

# Summarization using Sumy
def summarize_text(text, sentence_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# Topic Modeling using LDA
def extract_topics(text, num_topics=1, num_words=3):
    if not text or not text.strip():
        return ["No topic"]

    vectorizer = CountVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform([text])
        if X.shape[1] == 0:
            return ["No topic"]

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        topics = []
        for topic in lda.components_:
            words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-num_words - 1:-1]]
            topics.append(", ".join(words))
        return topics

    except Exception as e:
        print(f"⚠️ Topic extraction error: {e}")
        return ["No topic"]


# Main Enrichment Function
def enrich_articles(df):
    df['sentiment'] = df['summary'].fillna("").apply(analyze_sentiment)
    df['entities'] = df['text'].fillna("").apply(extract_entities)
    df['top_keywords'] = df['text'].fillna("").apply(lambda x: extract_top_keywords(x, 5))
    df['summary_auto'] = df['text'].fillna("").apply(lambda x: summarize_text(x, 2))
    df['topics'] = df['text'].fillna("").apply(lambda x: extract_topics(x, 1)[0])
    return df

# Save results
def save_enriched_data(df, path='data/enriched_articles.csv'):
    df.to_csv(path, index=False)
    print("✅ Enriched data saved to:", path)

if __name__ == "__main__":
    df = load_articles()
    df = enrich_articles(df)
    save_enriched_data(df)
