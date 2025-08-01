import streamlit as st
import pandas as pd
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import ast
import numpy as np
from scraper import scrape_rss, save_to_csv
from analyzer import load_articles, enrich_articles, save_enriched_data

st.set_page_config(page_title="🗞️ News Scraper & Analyzer", layout="wide")

st.title("🗞️ News Scraper & NLP Analyzer")
st.markdown("Scrape news from multiple sources and perform NLP analysis including sentiment, topic modeling, summarization, and keyword extraction.")

DATA_PATH = 'data/articles.csv'
ENRICHED_PATH = 'data/enriched_articles.csv'

# ------------------ Section 1: Scraping ------------------
st.header("🔄 Scrape News Articles")

if st.button("📡 Start Scraping"):
    with st.spinner("Scraping articles..."):
        articles = scrape_rss()
        if articles:
            save_to_csv(articles)
            st.success("✅ Scraping complete. Articles saved.")
        else:
            st.warning("⚠️ No articles were scraped.")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.subheader("📋 Scraped Articles Preview")
    st.dataframe(df[['title', 'source', 'published', 'url']], use_container_width=True)

    csv_raw = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Raw CSV", csv_raw, "articles.csv", "text/csv")
else:
    st.info("ℹ️ No scraped data found yet. Click above to start scraping.")

st.divider()

# ------------------ Section 2: NLP Analysis ------------------
st.header("🧠Analyze Articles with NLP")

if st.button("🧪 Run NLP Analysis"):
    if os.path.exists(DATA_PATH):
        with st.spinner("Analyzing articles..."):
            df = load_articles()
            enriched_df = enrich_articles(df)
            save_enriched_data(enriched_df)
            st.success("✅ Analysis complete. Enriched data saved.")

            enriched_csv = enriched_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Enriched CSV", enriched_csv, "enriched_articles.csv", "text/csv")
    else:
        st.error("❌ No scraped articles to analyze. Please scrape first.")

st.divider()

# ------------------ Section 3: Explore Articles ------------------
if os.path.exists(ENRICHED_PATH):
    st.header("Explore Articles")

    enriched_df = pd.read_csv(ENRICHED_PATH)

    with st.sidebar:
        st.header("🧰 Filters")
        selected_sources = st.multiselect("Select News Sources", options=enriched_df['source'].unique(), default=list(enriched_df['source'].unique()))
        sentiment_range = st.slider("Select Sentiment Range", min_value=-1.0, max_value=1.0, value=(-1.0, 1.0))

    filtered_df = enriched_df[
        (enriched_df['source'].isin(selected_sources)) &
        (enriched_df['sentiment'] >= sentiment_range[0]) &
        (enriched_df['sentiment'] <= sentiment_range[1])
    ]

    tab1, tab2 ,tab3= st.tabs(["📋 Article Table", "📊 Visualizations", "📰 Article Explorer"])

    with tab1:
        st.subheader("📋 Filtered Articles Table")
        st.dataframe(filtered_df[['title', 'source', 'sentiment', 'published']], use_container_width=True)

    with tab2:
        st.subheader("📈 Visual Insights")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown("#### 🥧 Top 10 Named Entities")
            if 'entities' in enriched_df.columns:
                all_entities = []
                for ents in enriched_df['entities'].dropna():
                    if isinstance(ents, str):
                        try:
                            ents_list = eval(ents) if ents.startswith("[") else ents.split(',')
                            all_entities.extend([e.strip() for e in ents_list if e.strip()])
                        except:
                            continue
                entity_counts = pd.Series(all_entities).value_counts().head(10)

                fig_entities, ax_entities = plt.subplots(figsize=(6, 6))  # Equal dimensions
                colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(entity_counts)))
                ax_entities.pie(
                    entity_counts,
                    labels=entity_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    wedgeprops={'edgecolor': 'white'}
                )
                ax_entities.axis('equal')
                fig_entities.tight_layout()
                st.pyplot(fig_entities)
            else:
                st.warning("No 'entities' column found in enriched data.")

        with col2:
            st.markdown("#### ☁️ Word Cloud of Keywords")
            if not enriched_df.empty:
                all_text = " ".join(enriched_df['text'].dropna().tolist())
                wordcloud = WordCloud(
                    width=600,
                    height=600,
                    background_color='white',
                    colormap='Reds'
                ).generate(all_text)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 6))  # Match pie size
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                fig_wc.tight_layout()
                st.pyplot(fig_wc)
            else:
                st.info("No text data to generate word cloud.")

        # Row 2: Avg Sentiment by Topic and Heatmap
        col3, col4 = st.columns(2, gap="medium")

        with col3:
            st.markdown("#### 📊 Average Sentiment by Topic")
            if not filtered_df.empty:
                avg_sentiment_topic = filtered_df.groupby('topics')['sentiment'].mean().reset_index()
                fig_sentiment = px.bar(
                    avg_sentiment_topic,
                    x='topics',
                    y='sentiment',
                    color='sentiment',
                    color_continuous_scale='reds',
                    title="Avg Sentiment per Topic",
                    height=400
                )
                fig_sentiment.update_layout(showlegend=False)
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.info("No data to display sentiment.")

        with col4:
            st.markdown("#### 🔥 Sentiment vs. Topic Heatmap")
            if not filtered_df.empty:
                heatmap_data = filtered_df.groupby(['topics', 'sentiment']).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(heatmap_data, annot=True, cmap="Reds", fmt="d", cbar=True, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Not enough data for heatmap.")

        
    with tab3:
        st.subheader("📰 Detailed Article Viewer")

        if not filtered_df.empty:
            selected_title = st.selectbox("Choose an Article", filtered_df['title'].tolist())
            article = filtered_df[filtered_df['title'] == selected_title].iloc[0]

            st.markdown(f"**📌 Title:** {article['title']}")
            st.markdown(f"**📰 Source:** {article['source']}")
            st.markdown(f"**📅 Published:** {article['published']}")
            st.markdown(f"**📈 Sentiment:** {round(article['sentiment'], 2)}")
            st.markdown(f"**🔑 Keywords:** {article['top_keywords']}")
            st.markdown(f"**🏷️ Entities:** {article['entities']}")
            st.markdown(f"**📚 Topics:** {article['topics']}")

            st.markdown("**📝 Summary:**")
            st.success(article['summary_auto'])

            with st.expander("📖 Show Full Text"):
                st.write(article['text'])
        else:
            st.warning("⚠️ No articles match the selected filters.")
else:
    st.info("ℹ️ No enriched articles found. Please complete the scraping and analysis steps.")

import streamlit as st
from qa_helper import answer_question  # assuming the function is here

st.markdown("## ❓ Question Answering on News Articles")

# Let the user select the article title
article_titles = enriched_df['title'].tolist()
selected_title = st.selectbox("Select an article:", article_titles)

# Retrieve content for selected article
selected_content = enriched_df[enriched_df['title'] == selected_title]['text'].values[0]

# Input question
question = st.text_input("Ask a question about the selected article:")

# Show answer when user asks
if question and st.button("Get Answer"):
    with st.spinner("Thinking..."):
        answer = answer_question(question, selected_content)
        st.success(f"**Answer:** {answer}")
with st.expander("📄 View Full Article"):
    st.write(selected_content)

# ------------------ Section 4: Translate Articles ------------------
from translator import translate_text, get_language_options

st.divider()
st.header("🌍 Translate Article")

if os.path.exists(ENRICHED_PATH):
    enriched_df = pd.read_csv(ENRICHED_PATH)
    if not enriched_df.empty:
        title_list = enriched_df['title'].dropna().tolist()
        selected_title = st.selectbox("📰 Choose an Article to Translate", title_list, key="translate_title")

        lang_options = get_language_options()
        selected_lang = st.selectbox("🌐 Choose Language", list(lang_options.keys()), key="translate_lang")
        selected_lang_code = lang_options[selected_lang]

        if st.button("🔁 Translate Article"):
            article_text = enriched_df[enriched_df['title'] == selected_title]['text'].values[0]

            if not isinstance(article_text, str) or not article_text.strip():
                st.error("❌ No article content found to translate.")
            else:
                with st.spinner("Translating..."):
                    translated_output = translate_text(article_text[:3000], selected_lang_code)  # Limit to avoid failure
                st.success(f"✅ Translated to {selected_lang}")
                st.text_area("📖 Translated Article", translated_output, height=300)

    else:
        st.warning("⚠️ No enriched articles found.")
else:
    st.info("ℹ️ Enriched file not found. Run analysis first.")


