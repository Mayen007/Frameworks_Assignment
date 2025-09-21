# Frameworks_Assignment

## Overview

This project explores the CORD-19 COVID-19 research dataset using Python. It covers data loading, cleaning, analysis, visualization, and an interactive Streamlit dashboard.

## Features

- Loads and cleans the CORD-19 metadata.csv dataset (1M+ papers)
- Handles missing data and inconsistent formats
- Extracts publication year, word counts, and other features
- Analyzes publication trends, top journals, and title word frequencies
- Visualizes data with matplotlib and wordcloud
- Interactive Streamlit app for data exploration

## Setup

1. Clone this repository and navigate to the project folder.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. **Download the CORD-19 `metadata.csv` file** from Kaggle (the file is too large to include in this repo):
   - [CORD-19 Research Challenge on Kaggle](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
   - Place `metadata.csv` in the project folder.

## Usage

### Run Data Analysis

```sh
python index.py
```

This will load, clean, and analyze the dataset, printing results and showing charts.

### Launch the Streamlit App

```sh
streamlit run app.py
```

This opens an interactive dashboard in your browser.

## Outputs

- Publication trends by year
- Top journals and sources
- Word cloud of paper titles
- Cleaned dataset and summary statistics

## Author

[Mayen007]

---

_See `REPORT.md` for a summary of findings and reflections._
