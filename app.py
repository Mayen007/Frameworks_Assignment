# app.py
# Streamlit App for CORD-19 Data Exploration
# Author: Mayen007
# Description: Interactive dashboard for exploring the CORD-19 COVID-19 research dataset.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set up Streamlit page configuration
st.set_page_config(
    page_title="CORD-19 COVID-19 Research Explorer", layout="wide")

# App title and description
st.title("CORD-19 COVID-19 Research Explorer")
st.markdown("""
This interactive app lets you explore the CORD-19 COVID-19 research dataset.\
Use the controls to filter by year and journal, view publication trends, top journals, word clouds, and more.
""")

# --- Load cleaned data ---


@st.cache_data
def load_data():
    # Loads cleaned DataFrame from index.py (runs cleaning logic if needed)
    import index  # This will run your cleaning code and create df_main_cleaned
    return index.df_main_cleaned.copy()


df = load_data()

# --- Sidebar controls ---
# Dropdowns for year and journal selection
years = sorted(df['publication_year'].dropna().unique())
# Limit to top 100 for dropdown
journals = ["All"] + sorted(df['journal'].dropna().unique()[:100])

selected_year = st.sidebar.selectbox("Select publication year", options=[
                                     "All"] + [int(y) for y in years], index=0)
selected_journal = st.sidebar.selectbox(
    "Select journal", options=journals, index=0)

# Filter DataFrame based on selections
filtered_df = df.copy()
if selected_year != "All":
    filtered_df = filtered_df[filtered_df['publication_year'] == int(
        selected_year)]
if selected_journal != "All":
    filtered_df = filtered_df[filtered_df['journal'] == selected_journal]

# --- Show sample of data ---
st.subheader("Sample of Filtered Data")
st.dataframe(filtered_df.head(20))

# --- Visualizations ---
# Publications by year
st.subheader("Number of Publications by Year")
year_counts = filtered_df['publication_year'].value_counts().sort_index()
fig1, ax1 = plt.subplots()
ax1.plot(year_counts.index, year_counts.values, marker='o')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Papers')
ax1.set_title('Number of Publications by Year')
st.pyplot(fig1)

# Top journals
st.subheader("Top 10 Journals Publishing COVID-19 Research")
journal_counts = filtered_df['journal'].value_counts().head(10)
fig2, ax2 = plt.subplots()
journal_counts.plot(kind='bar', ax=ax2)
ax2.set_xlabel('Journal')
ax2.set_ylabel('Number of Papers')
ax2.set_title('Top 10 Journals')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig2)

# Word cloud of titles
st.subheader("Word Cloud of Paper Titles")
title_words = filtered_df['title'].dropna().str.lower().str.replace(
    r'[^a-z ]', '', regex=True).str.cat(sep=' ')
if title_words:
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', max_words=100).generate(title_words)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)
else:
    st.info("No titles available for word cloud.")

# Top sources
st.subheader("Top 10 Sources of Papers")
source_counts = filtered_df['source_x'].value_counts().head(10)
fig4, ax4 = plt.subplots()
source_counts.plot(kind='bar', ax=ax4)
ax4.set_xlabel('Source')
ax4.set_ylabel('Number of Papers')
ax4.set_title('Top 10 Sources')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig4)

st.markdown("---")
st.caption("Built with Streamlit. Data: CORD-19 metadata.csv | Analysis: index.py")
