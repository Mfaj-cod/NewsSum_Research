# NewsSumm Dataset Cleaning Report

## 1. Objective

The goal was to clean and standardize the NewsSumm dataset before any model training.

## 2. Raw Dataset

- Original file: NewsSumm_Dataset.xlsx
- Original rows: 348766

## 3. Cleaning Steps Applied

1. Removed rows with missing article text or summary.
2. Removed HTML tags and markup.
3. Normalized whitespace and formatting.
4. Removed very short or corrupted entries.
5. Removed duplicate articles based on article_text.
6. Standardized column names.

## 4. Resulting Dataset

- Final cleaned rows: 307822
- Output file: NewsSumm_Cleaned.xlsx

## 5. Improvements Over Raw Version

- No duplicate articles
- No empty summaries
- Clean plain text without HTML or encoding noise
- More consistent formatting
- Ready for training abstractive summarization models

## 6. Notes

No engineered features were added at this stage. This step focuses purely on data quality and cleanliness.
