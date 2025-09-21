# CORD-19 COVID-19 Research Data Analysis Project

## Project Overview

This project explores the CORD-19 metadata.csv dataset, which contains over 1 million COVID-19 research papers. The workflow includes data loading, cleaning, analysis, visualization, and an interactive Streamlit app.

---

## Key Findings

- **Data Size:** 1,056,660 papers, 19 columns (raw)
- **Publication Trends:** Huge spike in 2020-2021, reflecting the global research response to COVID-19
- **Top Journals:** Identified the most prolific journals in COVID-19 research
- **Text Analysis:** Most frequent words in titles relate to COVID-19, health, and public response
- **Data Quality:** Some columns (e.g., `mag_id`) are 100% missing; most papers have titles and abstracts
- **Feature Engineering:** Added publication year, word counts, author counts, and text availability scores

---

## Visualizations

- **Publication Trends:** Line plot of papers per year
- **Top Journals:** Bar chart of top 10 journals
- **Word Cloud:** Most common words in paper titles
- **Source Distribution:** Bar chart of top 10 sources

---

## Streamlit App

- Interactive filters for year and journal
- Dynamic visualizations and data sample
- Easy exploration of trends and patterns

---

## Challenges & Reflections

- **Data Cleaning:** Handling inconsistent date formats and high-missing columns required careful logic
- **Performance:** Processing a large dataset (1M+ rows) was slow at times; caching and efficient filtering helped
- **Visualization:** Choosing the right charts and filtering stopwords for word clouds improved clarity
- **Learning:** Gained experience with pandas, matplotlib, wordcloud, and Streamlit for end-to-end data science workflows

---

## Next Steps

- Deeper text analysis (topic modeling, sentiment)
- More advanced interactive dashboards
- Exporting cleaned data for further research

---

## Author

[Mayen007]

---

_This project demonstrates a complete data science workflow, from raw data to interactive insights._
