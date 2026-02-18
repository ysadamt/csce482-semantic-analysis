# Reddit Sentiment Analysis Research Pipeline
## Methane/Dairy Climate Discourse Analysis

A reproducible, peer-review-ready research pipeline for analyzing public sentiment on climate-related discourse from Reddit.

## 📋 Project Overview

This project implements a 3-phase research methodology for social media sentiment analysis:

### Phase 1: Data Extraction
- Reddit API integration via PRAW
- Boolean query construction for construct validity
- Pagination support for temporal completeness
- User ID anonymization (SHA-256 hashing) for IRB compliance
- Query logging for reproducibility

### Phase 2: Data Preprocessing  
- Text cleaning (URLs, mentions, hashtags)
- Emoji handling (converts to descriptive text)
- SpaCy lemmatization
- Language detection filtering
- Length filtering (minimum 5 words)
- Deduplication
- Full attrition reporting

### Phase 3: Empirical Analysis
Three analysis techniques implemented:

1. **Trend-Based Temporal Analysis**
   - Monthly volume and sentiment trends
   - Rolling averages with confidence bands
   - First-half vs second-half statistical comparison

2. **Statistical Spike Detection**
   - σ-based anomaly detection (μ ± 2σ)
   - Volume and sentiment spike identification
   - Lag correlation analysis

3. **Odds Ratio / Log-Odds Analysis**
   - Semantic drivers of positive/negative sentiment
   - Probability distributions with 95% CI
   - Net sentiment calculation

## 🚀 Quick Start

### 1. Setup Reddit API Credentials

Create a `.env` file in the project root:

```bash
REDDIT_KEY=your_client_id
REDDIT_SECRET=your_client_secret
```

Get credentials at: https://www.reddit.com/prefs/apps (create a "script" app)

### 2. Install Dependencies

```bash
pip install -e .
# Or manually:
pip install emoji langdetect pyarrow spacy
python -m spacy download en_core_web_sm
```

### 3. Run the Pipeline

```bash
# Full pipeline (extract → preprocess → analyze)
python run_analysis.py --full

# Extract 2500 posts from 2018 onwards
python run_analysis.py --full --target 2500 --start-year 2018

# Analyze existing CSV file
python run_analysis.py --from-csv your_data.csv

# Run individual phases
python run_analysis.py --extract      # Extraction only
python run_analysis.py --preprocess   # Preprocessing only
python run_analysis.py --analyze      # Analysis only
```

## 📁 Project Structure

```
semantic-v2/
├── data_extractor.py    # Phase 1: Reddit data extraction
├── preprocessor.py      # Phase 2: Text preprocessing
├── analysis.py          # Phase 3: Sentiment analysis & visualization
├── run_analysis.py      # Main pipeline runner
├── model.py             # Legacy BERT model code
├── bert.py              # Legacy BERT training code
├── visualizer.py        # Legacy visualization code
├── data/                # Extracted and preprocessed data
└── results/             # Analysis outputs and figures
```

## 📊 Output Files

After running the pipeline, you'll find:

```
data/
├── reddit_extracted_TIMESTAMP.csv      # Raw extracted posts
├── reddit_preprocessed_TIMESTAMP.csv   # Cleaned posts
├── attrition_report_TIMESTAMP.csv      # Filtering statistics
└── query_log_TIMESTAMP.json            # Reproducibility log

results/
├── trend_analysis.png                  # Temporal trend visualization
├── spike_detection.png                 # Spike detection visualization
├── semantic_drivers.png                # Log-odds analysis visualization
├── probability_distribution.png        # Sentiment probability plot
├── analysis_results.json               # Complete numerical results
└── analyzed_data_TIMESTAMP.csv         # Final annotated dataset
```

## 📈 Analysis Outputs

### Attrition Table (Required for Publication)
```
Stage                  | Description                        | Count
-----------------------|------------------------------------|---------
N0_initial            | Initial dataset from extraction    | 3245
N1_deduplicated       | Removed exact duplicate texts      | 3012
N2_language_filtered  | Filtered to en language posts      | 2856
N3_length_filtered    | Removed posts with <5 words        | 2641
N4_final_analytic     | Final analytic dataset             | 2641
```

### Statistical Reporting Format
All results include:
- Test statistic
- p-value
- Effect size (Cohen's d)
- 95% confidence interval

Example:
```
"Net sentiment differed significantly between periods 
(Mann-Whitney U=842, p=0.003, Cohen's d=0.41)"
```

## 🔬 Methodology Notes

### Query Design
Queries follow Boolean logic framework:
```
(Primary Concept A OR Primary Concept B) 
AND (Contextual Constraint)
```

### Sentiment Model
Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` for classification:
- Negative (-1)
- Neutral (0)  
- Positive (+1)

### Spike Detection
Anomalies defined as observations exceeding:
- μ + 2σ (p < 0.05)
- μ + 2.58σ (p < 0.01)
- μ + 3.29σ (p < 0.001)

## 📚 Citation

If using this pipeline for research, please cite appropriate methodology:
- PRAW: Python Reddit API Wrapper
- HuggingFace Transformers
- CardiffNLP Twitter-RoBERTa

## 📄 License

MIT License
