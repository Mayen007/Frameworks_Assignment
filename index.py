# index.py
# CORD-19 Data Analysis Pipeline
# Author: [Mayen007]
# Description: Loads, cleans, analyzes, and visualizes the CORD-19 metadata.csv dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

print("Loading CORD-19 metadata.csv file...")

# Load the metadata.csv file into a pandas DataFrame
# Handles missing file and other errors gracefully
try:
    df = pd.read_csv('metadata.csv')
    print(f"Successfully loaded {len(df):,} records from metadata.csv\n")
except FileNotFoundError:
    print("Error: metadata.csv file not found!")
    exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Part 1: Data Loading and Basic Exploration
# ------------------------------------------
# Shows first rows, column names, dimensions, data types, missing values, and basic stats

print("=== CORD-19 Dataset Analysis ===\n")

# 1. Examine the first few rows and data structure
print("1. First 5 rows of the dataset:")
print(df.head())
print("\n" + "="*50 + "\n")

print("2. Column names:")
print(df.columns.tolist())
print("\n" + "="*50 + "\n")

# 3. DataFrame dimensions
print("3. Dataset dimensions:")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print("\n" + "="*50 + "\n")

# 4. Data types
print("4. Data types of each column:")
print(df.dtypes)
print("\n" + "="*50 + "\n")

# 5. Missing values analysis
print("5. Missing values analysis:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing Percentage': missing_percentage.values
})
missing_df = missing_df[missing_df['Missing Count'] >
                        0].sort_values('Missing Count', ascending=False)
print(missing_df.to_string(index=False))
print("\n" + "="*50 + "\n")

# 6. Basic statistics for numerical columns
print("6. Basic statistics for numerical columns:")
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
if len(numerical_columns) > 0:
    print(df[numerical_columns].describe())
else:
    print("No numerical columns found in the dataset")
print("\n" + "="*50 + "\n")

# 7. Publication year analysis (handling inconsistent date formats)
print("7. Publication year analysis:")
try:
    # Convert publish_time to datetime with error handling for inconsistent formats
    df['publish_time_clean'] = pd.to_datetime(
        df['publish_time'], errors='coerce')
    df['year'] = df['publish_time_clean'].dt.year

    # Show year distribution
    year_counts = df['year'].value_counts().sort_index()
    print("Publications by year (top 10):")
    print(year_counts.head(10))

    # Show some examples of problematic dates
    problematic_dates = df[df['publish_time_clean'].isnull(
    ) & df['publish_time'].notna()]
    if len(problematic_dates) > 0:
        print(
            f"\nFound {len(problematic_dates)} rows with unparseable dates. Examples:")
        print(problematic_dates[['publish_time']
                                ].head().to_string(index=False))

except Exception as e:
    print(f"Error in date analysis: {e}")

print("\n" + "="*50 + "\n")

# 8. Key insights summary
print("8. Key Dataset Insights:")
print(f"• Total papers: {len(df):,}")
print(f"• Total columns: {df.shape[1]}")
print(
    f"• Most complete column: {missing_df.iloc[-1]['Column'] if len(missing_df) > 0 else 'All columns complete'}")
print(
    f"• Column with most missing data: {missing_df.iloc[0]['Column'] if len(missing_df) > 0 else 'No missing data'}")
if 'year' in df.columns:
    valid_years = df['year'].dropna()
    if len(valid_years) > 0:
        print(
            f"• Publication years range: {valid_years.min():.0f} to {valid_years.max():.0f}")
        print(f"• Peak publication year: {valid_years.mode().iloc[0]:.0f}")

print("\nData loading and basic exploration completed!")

print("\n" + "="*60)
print("PART 2: DATA CLEANING AND PREPARATION")
print("="*60 + "\n")

# ==========================================
# 1. IDENTIFY COLUMNS WITH MANY MISSING VALUES
# ==========================================
print("1. MISSING VALUES ANALYSIS FOR CLEANING DECISIONS")
print("-" * 50)

# Create detailed missing values analysis
missing_analysis = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
    'Data_Type': df.dtypes,
    'Non_Null_Count': df.count()
})
missing_analysis = missing_analysis.sort_values(
    'Missing_Percentage', ascending=False)

print("Missing values summary (sorted by percentage):")
print(missing_analysis.to_string(index=False))

# Categorize columns by missing data severity
high_missing = missing_analysis[missing_analysis['Missing_Percentage'] > 50]
medium_missing = missing_analysis[(missing_analysis['Missing_Percentage'] > 10) &
                                  (missing_analysis['Missing_Percentage'] <= 50)]
low_missing = missing_analysis[missing_analysis['Missing_Percentage'] <= 10]

print(f"\nCategorization by missing data:")
print(
    f"• High missing (>50%): {len(high_missing)} columns - {list(high_missing['Column'])}")
print(
    f"• Medium missing (10-50%): {len(medium_missing)} columns - {list(medium_missing['Column'])}")
print(
    f"• Low missing (≤10%): {len(low_missing)} columns - {list(low_missing['Column'])}")

print("\n" + "="*50 + "\n")

# ==========================================
# 2. DECIDE HOW TO HANDLE MISSING VALUES
# ==========================================
print("2. MISSING DATA HANDLING STRATEGY")
print("-" * 50)

# Define cleaning strategy based on data analysis
columns_to_drop = ['mag_id']  # 100% missing - completely useless
# High missing but might be useful for filtering
columns_to_keep_with_nulls = ['arxiv_id', 'who_covidence_id']
columns_for_analysis = ['cord_uid', 'title', 'abstract',
                        'publish_time', 'authors', 'journal', 'doi', 'pmcid', 'pubmed_id']

print("Cleaning strategy:")
print(f"• DROP completely: {columns_to_drop} (100% missing)")
print(
    f"• KEEP with nulls: {columns_to_keep_with_nulls} (useful for filtering specific subsets)")
print(f"• CORE analysis columns: {columns_for_analysis}")

# Check what we'll lose if we drop rows with missing core data
core_columns_missing = df[columns_for_analysis].isnull().sum()
print(f"\nMissing values in core analysis columns:")
for col in columns_for_analysis:
    missing_count = core_columns_missing[col]
    missing_pct = (missing_count / len(df)) * 100
    print(f"  {col}: {missing_count:,} ({missing_pct:.2f}%)")

# Create subsets for different analysis needs
print(f"\nDataset subset options:")
complete_core_mask = df[columns_for_analysis].notna().all(axis=1)
print(
    f"• Complete core data: {complete_core_mask.sum():,} rows ({(complete_core_mask.sum()/len(df)*100):.1f}%)")

has_abstract_mask = df['abstract'].notna()
print(
    f"• Has abstract: {has_abstract_mask.sum():,} rows ({(has_abstract_mask.sum()/len(df)*100):.1f}%)")

has_title_mask = df['title'].notna()
print(
    f"• Has title: {has_title_mask.sum():,} rows ({(has_title_mask.sum()/len(df)*100):.1f}%)")

print("\n" + "="*50 + "\n")

# ==========================================
# 3. CREATE CLEANED VERSION OF DATASET
# ==========================================
print("3. CREATING CLEANED DATASET")
print("-" * 50)

# Create cleaned dataset
print("Creating cleaned dataset...")

# Step 1: Drop completely useless columns
df_cleaned = df.drop(columns=columns_to_drop)
print(f"• Dropped {len(columns_to_drop)} useless columns")

# Step 2: Create different versions for different analysis needs

# Version 1: Keep all rows, just clean columns
df_all_rows = df_cleaned.copy()
print(
    f"• Version 1 (all rows): {len(df_all_rows):,} rows × {df_all_rows.shape[1]} columns")

# Version 2: Only rows with titles (minimal requirement)
df_with_titles = df_cleaned[df_cleaned['title'].notna()].copy()
print(
    f"• Version 2 (has title): {len(df_with_titles):,} rows × {df_with_titles.shape[1]} columns")

# Version 3: Rows with abstracts (good for text analysis)
df_with_abstracts = df_cleaned[df_cleaned['abstract'].notna()].copy()
print(
    f"• Version 3 (has abstract): {len(df_with_abstracts):,} rows × {df_with_abstracts.shape[1]} columns")

# Version 4: Complete core data (most restrictive)
core_complete_mask = df_cleaned[columns_for_analysis].notna().all(axis=1)
df_complete_core = df_cleaned[core_complete_mask].copy()
print(
    f"• Version 4 (complete core): {len(df_complete_core):,} rows × {df_complete_core.shape[1]} columns")

# Choose the main cleaned dataset (balance between data retention and completeness)
# Good balance - keeps most data but ensures we have titles
df_main_cleaned = df_with_titles.copy()
print(
    f"\nSelected main cleaned dataset: {len(df_main_cleaned):,} rows (Version 2)")

print("\n" + "="*50 + "\n")

# ==========================================
# 4. CONVERT DATE COLUMNS TO DATETIME FORMAT
# ==========================================
print("4. CONVERTING DATE COLUMNS")
print("-" * 50)

print("Converting publish_time to proper datetime format...")

# Analyze the date formats first
print("Sample publish_time values:")
sample_dates = df_main_cleaned['publish_time'].dropna().head(10)
for i, date_val in enumerate(sample_dates):
    print(f"  {i+1}. '{date_val}'")

# Convert with comprehensive error handling


def convert_publish_time(date_str):
    """Convert various date formats to datetime"""
    if pd.isna(date_str):
        return pd.NaT

    date_str = str(date_str).strip()

    # Try different date formats
    formats_to_try = [
        '%Y-%m-%d',      # 2020-01-15
        '%Y/%m/%d',      # 2020/01/15
        '%m/%d/%Y',      # 01/15/2020
        '%Y-%m',         # 2020-01
        '%Y',            # 2020
        '%b %d, %Y',     # Jan 15, 2020
        '%B %d, %Y',     # January 15, 2020
        '%d %b %Y',      # 15 Jan 2020
    ]

    for fmt in formats_to_try:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue

    # If all else fails, try pandas' flexible parser
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT


# Apply the conversion
print("Applying datetime conversion...")
df_main_cleaned['publish_time_dt'] = df_main_cleaned['publish_time'].apply(
    convert_publish_time)

# Check conversion results
total_dates = len(df_main_cleaned[df_main_cleaned['publish_time'].notna()])
converted_dates = len(
    df_main_cleaned[df_main_cleaned['publish_time_dt'].notna()])
failed_conversions = total_dates - converted_dates

print(f"Conversion results:")
print(f"• Total non-null dates: {total_dates:,}")
print(
    f"• Successfully converted: {converted_dates:,} ({(converted_dates/total_dates*100):.1f}%)")
print(
    f"• Failed conversions: {failed_conversions:,} ({(failed_conversions/total_dates*100):.1f}%)")

if failed_conversions > 0:
    print(f"\nSample failed conversions:")
    failed_dates = df_main_cleaned[df_main_cleaned['publish_time_dt'].isna() &
                                   df_main_cleaned['publish_time'].notna()]['publish_time'].head(5)
    for date_val in failed_dates:
        print(f"  '{date_val}'")

print("\n" + "="*50 + "\n")

# ==========================================
# 5. EXTRACT YEAR FROM PUBLICATION DATE
# ==========================================
print("5. EXTRACTING PUBLICATION YEAR")
print("-" * 50)

# Extract year from the converted datetime
df_main_cleaned['publication_year'] = df_main_cleaned['publish_time_dt'].dt.year

# Analyze year distribution
valid_years = df_main_cleaned['publication_year'].dropna()
print(f"Year extraction results:")
print(f"• Total papers with valid years: {len(valid_years):,}")
print(f"• Year range: {valid_years.min():.0f} to {valid_years.max():.0f}")
print(f"• Most common year: {valid_years.mode().iloc[0]:.0f}")

# Show year distribution for recent years
print(f"\nPublication distribution (last 10 years):")
recent_years = valid_years[valid_years >= 2014].value_counts().sort_index()
for year, count in recent_years.items():
    print(f"  {year:.0f}: {count:,} papers")

print("\n" + "="*50 + "\n")

# ==========================================
# 6. CREATE NEW ANALYTICAL COLUMNS
# ==========================================
print("6. CREATING NEW ANALYTICAL COLUMNS")
print("-" * 50)

print("Creating derived features...")

# 1. Abstract word count


def count_words(text):
    """Count words in text, handling NaN values"""
    if pd.isna(text):
        return 0
    return len(str(text).split())


df_main_cleaned['abstract_word_count'] = df_main_cleaned['abstract'].apply(
    count_words)
print(
    f"• Abstract word count: avg {df_main_cleaned['abstract_word_count'].mean():.1f} words")

# 2. Title word count
df_main_cleaned['title_word_count'] = df_main_cleaned['title'].apply(
    count_words)
print(
    f"• Title word count: avg {df_main_cleaned['title_word_count'].mean():.1f} words")

# 3. Author count (rough estimate)


def count_authors(author_str):
    """Estimate number of authors from author string"""
    if pd.isna(author_str):
        return 0
    # Simple heuristic: count semicolons + 1, or commas + 1
    author_str = str(author_str)
    if ';' in author_str:
        return len(author_str.split(';'))
    elif ',' in author_str:
        # More conservative estimate for comma-separated
        # Assume firstname, lastname pattern
        return len(author_str.split(',')) // 2
    else:
        return 1 if author_str.strip() else 0


df_main_cleaned['author_count'] = df_main_cleaned['authors'].apply(
    count_authors)
print(
    f"• Author count: avg {df_main_cleaned['author_count'].mean():.1f} authors")

# 4. Has DOI flag
df_main_cleaned['has_doi'] = df_main_cleaned['doi'].notna()
print(
    f"• Papers with DOI: {df_main_cleaned['has_doi'].sum():,} ({df_main_cleaned['has_doi'].mean()*100:.1f}%)")

# 5. Has PMC ID flag
df_main_cleaned['has_pmcid'] = df_main_cleaned['pmcid'].notna()
print(
    f"• Papers with PMC ID: {df_main_cleaned['has_pmcid'].sum():,} ({df_main_cleaned['has_pmcid'].mean()*100:.1f}%)")

# 6. Publication recency (years since publication)
current_year = 2024
df_main_cleaned['years_since_publication'] = current_year - \
    df_main_cleaned['publication_year']
print(
    f"• Average years since publication: {df_main_cleaned['years_since_publication'].mean():.1f} years")

# 7. Text availability score (0-3 based on available text fields)


def text_availability_score(row):
    """Calculate text availability score based on title, abstract, etc."""
    score = 0
    if pd.notna(row['title']) and len(str(row['title']).strip()) > 0:
        score += 1
    if pd.notna(row['abstract']) and len(str(row['abstract']).strip()) > 0:
        score += 2  # Abstract is more valuable
    return score


df_main_cleaned['text_availability_score'] = df_main_cleaned.apply(
    text_availability_score, axis=1)
score_dist = df_main_cleaned['text_availability_score'].value_counts(
).sort_index()
print(f"• Text availability distribution:")
for score, count in score_dist.items():
    print(
        f"    Score {score}: {count:,} papers ({count/len(df_main_cleaned)*100:.1f}%)")

print("\n" + "="*50 + "\n")

# ==========================================
# 7. VALIDATE CLEANED DATASET
# ==========================================
print("7. CLEANED DATASET VALIDATION & SUMMARY")
print("-" * 50)

print("Final cleaned dataset summary:")
print(f"• Original dataset: {len(df):,} rows × {df.shape[1]} columns")
print(
    f"• Cleaned dataset: {len(df_main_cleaned):,} rows × {df_main_cleaned.shape[1]} columns")
print(f"• Data retention: {(len(df_main_cleaned)/len(df)*100):.1f}%")

# Show new columns created
new_columns = ['publish_time_dt', 'publication_year', 'abstract_word_count', 'title_word_count',
               'author_count', 'has_doi', 'has_pmcid', 'years_since_publication', 'text_availability_score']
print(f"\nNew analytical columns created: {len(new_columns)}")
for col in new_columns:
    if col in df_main_cleaned.columns:
        print(f"  ✓ {col}")

# Quick quality checks
print(f"\nData quality checks:")
print(
    f"• Rows with valid titles: {df_main_cleaned['title'].notna().sum():,} (100.0%)")
print(
    f"• Rows with abstracts: {df_main_cleaned['abstract'].notna().sum():,} ({df_main_cleaned['abstract'].notna().mean()*100:.1f}%)")
print(
    f"• Rows with valid publication years: {df_main_cleaned['publication_year'].notna().sum():,} ({df_main_cleaned['publication_year'].notna().mean()*100:.1f}%)")
print(
    f"• Average text availability score: {df_main_cleaned['text_availability_score'].mean():.2f}/3")

# Show descriptive statistics for new numerical columns
print(f"\nDescriptive statistics for new numerical features:")
numerical_features = ['abstract_word_count', 'title_word_count',
                      'author_count', 'publication_year', 'years_since_publication']
stats_df = df_main_cleaned[numerical_features].describe()
print(stats_df.round(2))

# Save cleaned dataset (optional)
print(f"\nCleaning process completed successfully!")
print(f"Main cleaned dataset ready for analysis: 'df_main_cleaned'")
print(f"Alternative datasets available: 'df_with_abstracts', 'df_complete_core'")

print("\n" + "="*60)
print("DATA CLEANING AND PREPARATION COMPLETED!")
print("="*60)

# =============================================================
# PART 3: DATA ANALYSIS AND VISUALIZATION
# =============================================================

try:
    from wordcloud import WordCloud
    wordcloud_available = True
except ImportError:
    wordcloud_available = False
    print("[INFO] 'wordcloud' package not installed. Word cloud will be skipped.")

print("\n" + "="*60)
print("PART 3: DATA ANALYSIS AND VISUALIZATION")
print("="*60 + "\n")

# 1. Count papers by publication year
print("1. Papers by publication year:")
year_counts = df_main_cleaned['publication_year'].value_counts().sort_index()
print(year_counts.tail(15))

# 2. Identify top journals
print("\n2. Top 10 journals publishing COVID-19 research:")
journal_counts = df_main_cleaned['journal'].value_counts().head(10)
print(journal_counts)

# 3. Most frequent words in titles
print("\n3. Most frequent words in paper titles:")
title_words = df_main_cleaned['title'].dropna().str.lower(
).str.replace(r'[^a-z ]', '', regex=True).str.split()
all_words = [word for words in title_words for word in words if len(word) > 2 and word not in {'the', 'and', 'for', 'with', 'from', 'are', 'that', 'this', 'was', 'use', 'using', 'can', 'covid', 'covid19', 'sarscov2', 'coronavirus', 'study', 'data', 'analysis', 'based', 'case', 'new', 'results', 'effect', 'effects', 'between', 'after', 'during', 'among', 'patients', 'disease', 'infection',
                                                                                               'response', 'risk', 'role', 'associated', 'report', 'review', 'systematic', 'meta', 'case', 'series', 'case', 'report', 'reports', 'study', 'studies', 'review', 'reviews', 'systematic', 'meta', 'analysis', 'analyses', 'evidence', 'impact', 'outcomes', 'clinical', 'health', 'public', 'pandemic', 'time', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}]
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(20)
print("Top 20 words:")
for word, freq in most_common_words:
    print(f"  {word}: {freq}")

# 4. Plot number of publications over time
plt.figure(figsize=(10, 5))
plt.plot(year_counts.index, year_counts.values, marker='o')
plt.title('Number of Publications by Year')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Bar chart of top publishing journals
plt.figure(figsize=(10, 6))
journal_counts.plot(kind='bar')
plt.title('Top 10 Journals Publishing COVID-19 Research')
plt.xlabel('Journal')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 6. Word cloud of paper titles
if wordcloud_available:
    print("\nGenerating word cloud for paper titles...")
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          max_words=100).generate(' '.join(all_words))
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Paper Titles')
    plt.tight_layout()
    plt.show()
else:
    print("[INFO] Skipping word cloud (wordcloud package not installed)")

# 7. Plot distribution of paper counts by source
print("\n7. Distribution of paper counts by source:")
source_counts = df_main_cleaned['source_x'].value_counts().head(10)
print(source_counts)
plt.figure(figsize=(10, 6))
source_counts.plot(kind='bar')
plt.title('Top 10 Sources of Papers')
plt.xlabel('Source')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nData analysis and visualization completed!\n")
# =============================================================
# END OF PART 3
# =============================================================
