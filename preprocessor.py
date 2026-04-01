"""
Phase 2: Data Preprocessing
Cleaning Without Destroying Meaning

This module provides a comprehensive preprocessing pipeline with:
- Text cleaning (URLs, mentions, hashtags)
- Emoji handling (convert to text)
- Linguistic normalization (lemmatization via SpaCy)
- Quality control (length filtering, deduplication, language detection)
- Full attrition reporting for publication
"""

import pandas as pd
import numpy as np
import re
import emoji
import spacy
from typing import List, Dict, Tuple, Optional
from collections import Counter
import hashlib

# Try to import language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect not installed. Run: pip install langdetect")


class DataPreprocessor:
    """
    Comprehensive text preprocessing with full audit trail.
    
    Follows best practices:
    - Preserves raw_text for reproducibility
    - Handles emojis intentionally (converts to text)
    - Uses SpaCy for lemmatization
    - Tracks attrition at each stage
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize preprocessor with SpaCy model."""
        try:
            self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
        except OSError:
            print(f"Downloading SpaCy model: {spacy_model}")
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
        
        # Increase max length for long texts
        self.nlp.max_length = 2000000
        
        self.attrition_log = {}
        self.processing_log = []
        
        # Custom stopwords for domain
        self.custom_stopwords = {
            'http', 'https', 'www', 'com', 'org', 'amp',
            'deleted', 'removed', 'click', 'link', 'edit',
            'reddit', 'subreddit', 'post', 'comment'
        }
    
    def _log_stage(self, stage: str, count_before: int, count_after: int, 
                   description: str = ""):
        """Log preprocessing stage for attrition report."""
        self.attrition_log[stage] = {
            "count_before": count_before,
            "count_after": count_after,
            "removed": count_before - count_after,
            "removal_rate": (count_before - count_after) / count_before * 100 if count_before > 0 else 0,
            "description": description
        }
        self.processing_log.append({
            "stage": stage,
            "count": count_after,
            "timestamp": pd.Timestamp.now().isoformat()
        })
    
    def convert_emojis_to_text(self, text: str) -> str:
        """
        Convert emojis to descriptive text.
        
        This preserves emotional signal that would be lost with blind removal.
        Example: 😡 → angry_face
        """
        return emoji.demojize(text, delimiters=(" ", " "))
    
    def clean_text(self, text: str, preserve_hashtag_text: bool = True) -> str:
        """
        Apply controlled text cleaning pipeline.
        
        Steps:
        1. Convert emojis to text (preserve emotional signal)
        2. Remove URLs
        3. Remove @mentions
        4. Handle hashtags (strip # but keep word if specified)
        5. Normalize whitespace
        6. Remove special characters
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Step 1: Convert emojis to text
        text = self.convert_emojis_to_text(text)
        
        # Step 2: Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Step 3: Remove @mentions (privacy + noise)
        text = re.sub(r'@\w+', '', text)
        
        # Step 4: Handle hashtags
        if preserve_hashtag_text:
            # Strip # but keep the word (topic signal)
            text = re.sub(r'#(\w+)', r'\1', text)
        else:
            text = re.sub(r'#\w+', '', text)
        
        # Step 5: Remove special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Step 6: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def lemmatize_text(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Apply SpaCy lemmatization.
        
        Note: For BERT models, stopwords are often retained.
        For topic modeling, they may be removed.
        """
        if not text:
            return ""
        
        doc = self.nlp(text.lower())
        
        lemmas = []
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
            
            # Optionally skip stopwords
            if remove_stopwords and (token.is_stop or 
                                     token.text in self.custom_stopwords):
                continue
            
            lemmas.append(token.lemma_)
        
        return " ".join(lemmas)
    
    def get_tokens(self, text: str, remove_stopwords: bool = True,
                   min_length: int = 2) -> List[str]:
        """Extract tokens for classical NLP analysis."""
        if not text:
            return []
        
        doc = self.nlp(text.lower())
        
        tokens = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if remove_stopwords and (token.is_stop or 
                                     token.text in self.custom_stopwords):
                continue
            if len(token.text) < min_length:
                continue
            
            tokens.append(token.lemma_)
        
        return tokens
    
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        if not LANGDETECT_AVAILABLE:
            return "unknown"
        
        try:
            if len(text.strip()) < 10:
                return "unknown"
            return detect(text)
        except:
            return "unknown"
    
    def preprocess_dataframe(self, df: pd.DataFrame,
                            text_column: str = 'raw_text',
                            comments_column: str = 'comments_text',
                            include_comments_in_text: bool = True,
                            min_word_count: int = 5,
                            language: str = 'en',
                            remove_duplicates: bool = True,
                            lemmatize_for_bert: bool = False,
                            lemmatize_for_topics: bool = True) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing raw text
            comments_column: Name of optional column containing scraped comments
            include_comments_in_text: Merge comments into preprocessing/analysis text
            min_word_count: Minimum words required (default: 5)
            language: Language to filter for (default: 'en')
            remove_duplicates: Whether to remove duplicate texts
            lemmatize_for_bert: Apply lemmatization for BERT (usually False)
            lemmatize_for_topics: Create lemmatized version for topic modeling
        
        Returns:
            Preprocessed DataFrame with all columns preserved
        """
        print(f"\n{'='*60}")
        print("DATA PREPROCESSING - PHASE 2")
        print(f"{'='*60}\n")
        
        df = df.copy()
        initial_count = len(df)
        self._log_stage("N0_initial", initial_count, initial_count, 
                       "Initial dataset from extraction")
        print(f"📥 Initial dataset: {initial_count} posts")

        # Build an explicit analysis text source to keep raw_text reproducible.
        df[text_column] = df[text_column].fillna('').astype(str)
        effective_text_column = text_column
        if include_comments_in_text and comments_column in df.columns:
            df[comments_column] = df[comments_column].fillna('').astype(str)
            df['analysis_text'] = (
                df[text_column].str.strip() + " " + df[comments_column].str.strip()
            ).str.strip()
            effective_text_column = 'analysis_text'
            print("💬 Including comments in analysis text")
        else:
            print("📝 Using post text only for analysis")
        
        # ===== STEP 1: Remove exact duplicates =====
        if remove_duplicates:
            count_before = len(df)
            df = df.drop_duplicates(subset=[effective_text_column], keep='first')
            count_after = len(df)
            self._log_stage("N1_deduplicated", count_before, count_after,
                          "Removed exact duplicate texts")
            print(f"🔄 After deduplication: {count_after} posts "
                  f"(-{count_before - count_after})")
        
        # ===== STEP 2: Language filtering =====
        if LANGDETECT_AVAILABLE and language:
            count_before = len(df)
            print(f"🌐 Detecting language (this may take a moment)...")
            df['detected_language'] = df[effective_text_column].apply(self.detect_language)
            df = df[df['detected_language'].isin([language, 'unknown'])]
            count_after = len(df)
            self._log_stage("N2_language_filtered", count_before, count_after,
                          f"Filtered to {language} language posts")
            print(f"🌐 After language filter ({language}): {count_after} posts "
                  f"(-{count_before - count_after})")
        
        # ===== STEP 3: Text cleaning =====
        print(f"🧹 Cleaning text...")
        df['clean_text'] = df[effective_text_column].apply(self.clean_text)
        
        # ===== STEP 4: Length filtering =====
        count_before = len(df)
        df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        df = df[df['word_count'] >= min_word_count]
        count_after = len(df)
        self._log_stage("N3_length_filtered", count_before, count_after,
                       f"Removed posts with <{min_word_count} words")
        print(f"📏 After length filter (≥{min_word_count} words): {count_after} posts "
              f"(-{count_before - count_after})")
        
        # ===== STEP 5: Lemmatization (BATCH PROCESSING for speed) =====
        if lemmatize_for_bert:
            print(f"📝 Lemmatizing for BERT (batch processing)...")
            texts = df['clean_text'].tolist()
            lemmatized = []
            for i, doc in enumerate(self.nlp.pipe(texts, batch_size=100, n_process=1)):
                lemmas = [token.lemma_ for token in doc 
                         if not token.is_punct and not token.is_space]
                lemmatized.append(" ".join(lemmas))
                if (i + 1) % 500 == 0:
                    print(f"   Processed {i + 1}/{len(texts)} texts...")
            df['clean_text_lemma'] = lemmatized
        
        if lemmatize_for_topics:
            print(f"📝 Lemmatizing for topic modeling (batch processing)...")
            texts = df['clean_text'].str.lower().tolist()
            topic_texts = []
            token_lists = []
            for i, doc in enumerate(self.nlp.pipe(texts, batch_size=100, n_process=1)):
                # For topic text (with stopword removal)
                lemmas = [token.lemma_ for token in doc 
                         if not token.is_punct and not token.is_space
                         and not token.is_stop 
                         and token.text not in self.custom_stopwords
                         and len(token.text) >= 2]
                topic_texts.append(" ".join(lemmas))
                token_lists.append(lemmas)
                if (i + 1) % 500 == 0:
                    print(f"   Processed {i + 1}/{len(texts)} texts...")
            df['topic_text'] = topic_texts
            df['tokens'] = token_lists
        
        # ===== STEP 6: Final validation =====
        count_before = len(df)
        # Remove any rows where cleaning resulted in empty text
        df = df[df['clean_text'].str.len() > 0]
        count_after = len(df)
        self._log_stage("N4_final_analytic", count_before, count_after,
                       "Final analytic dataset after all preprocessing")
        
        print(f"\n{'='*60}")
        print(f"✅ PREPROCESSING COMPLETE")
        print(f"   Initial posts: {initial_count}")
        print(f"   Final posts: {count_after}")
        print(f"   Total removed: {initial_count - count_after} "
              f"({(initial_count - count_after) / initial_count * 100:.1f}%)")
        print(f"{'='*60}\n")
        
        return df
    
    def get_attrition_table(self) -> pd.DataFrame:
        """
        Generate attrition table for publication.
        
        Required format per methodology:
        Stage | Description | Tweet/Post Count
        """
        rows = []
        for stage, info in self.attrition_log.items():
            rows.append({
                "Stage": stage,
                "Description": info.get('description', ''),
                "Count": info.get('count_after', 0),
                "Removed": info.get('removed', 0),
                "Removal Rate (%)": round(info.get('removal_rate', 0), 2)
            })
        
        return pd.DataFrame(rows)
    
    def compute_text_statistics(self, df: pd.DataFrame, 
                                text_column: str = 'clean_text') -> Dict:
        """Compute descriptive statistics for text corpus."""
        stats = {
            "total_posts": len(df),
            "avg_word_count": df['word_count'].mean(),
            "median_word_count": df['word_count'].median(),
            "std_word_count": df['word_count'].std(),
            "min_word_count": df['word_count'].min(),
            "max_word_count": df['word_count'].max(),
        }
        
        if 'tokens' in df.columns:
            all_tokens = [token for tokens in df['tokens'] for token in tokens]
            token_freq = Counter(all_tokens)
            stats["unique_tokens"] = len(token_freq)
            stats["total_tokens"] = len(all_tokens)
            stats["top_20_tokens"] = dict(token_freq.most_common(20))
        
        return stats
    
    def save_preprocessed_data(self, df: pd.DataFrame,
                               output_path: str,
                               save_attrition: bool = True) -> Dict[str, str]:
        """Save preprocessed data and attrition report."""
        import os
        import json
        
        paths = {}
        
        # Save main data
        if output_path.endswith('.parquet'):
            # Convert list columns to strings for parquet
            df_save = df.copy()
            if 'tokens' in df_save.columns:
                df_save['tokens'] = df_save['tokens'].apply(
                    lambda x: '|'.join(x) if isinstance(x, list) else x
                )
            df_save.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        paths['data'] = output_path
        
        if save_attrition:
            # Save attrition table
            base_path = output_path.rsplit('.', 1)[0]
            attrition_path = f"{base_path}_attrition.csv"
            attrition_df = self.get_attrition_table()
            attrition_df.to_csv(attrition_path, index=False)
            paths['attrition'] = attrition_path
            
            # Save text statistics
            stats_path = f"{base_path}_stats.json"
            stats = self.compute_text_statistics(df)
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            paths['statistics'] = stats_path
        
        return paths


def preprocess_reddit_data(df: pd.DataFrame,
                          output_path: str = None,
                          text_column: str = 'raw_text',
                          comments_column: str = 'comments_text',
                          include_comments_in_text: bool = True) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Convenience function to run full preprocessing pipeline.
    
    Args:
        df: DataFrame from data extraction
        output_path: Optional path to save preprocessed data
        text_column: Name of post text column
        comments_column: Name of comments text column
        include_comments_in_text: Merge comments into analysis text if available
        
    Returns:
        Tuple of (preprocessed DataFrame, preprocessor object)
    """
    preprocessor = DataPreprocessor()
    
    # Run preprocessing
    df_clean = preprocessor.preprocess_dataframe(
        df,
        text_column=text_column,
        comments_column=comments_column,
        include_comments_in_text=include_comments_in_text,
        min_word_count=5,
        language='en',
        remove_duplicates=True,
        lemmatize_for_bert=False,  # BERT handles its own tokenization
        lemmatize_for_topics=True
    )
    
    # Print attrition table
    print("\n📊 ATTRITION TABLE (Required for Publication):")
    print(preprocessor.get_attrition_table().to_string(index=False))
    
    # Print text statistics
    print("\n📈 TEXT CORPUS STATISTICS:")
    stats = preprocessor.compute_text_statistics(df_clean)
    for key, value in stats.items():
        if key != 'top_20_tokens':
            print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
    
    print("\n   Top 20 tokens:")
    for token, count in stats.get('top_20_tokens', {}).items():
        print(f"      {token}: {count}")
    
    # Save if path provided
    if output_path:
        paths = preprocessor.save_preprocessed_data(df_clean, output_path)
        print(f"\n💾 Saved preprocessed data to: {paths}")
    
    return df_clean, preprocessor


if __name__ == "__main__":
    # Example usage with mock data
    mock_data = pd.DataFrame({
        'raw_text': [
            "Methane from dairy cows is a major greenhouse gas problem! 😡 https://example.com",
            "New anaerobic digesters are reducing emissions on farms 🌱",
            "Climate change is real and we need to act now @JohnDoe",
            "Short",  # Will be filtered
            "RT RT RT",  # Will be filtered
            "Cows produce methane through enteric fermentation, contributing to global warming",
        ],
        'created_utc': pd.date_range('2023-01-01', periods=6, freq='D')
    })
    
    df_clean, preprocessor = preprocess_reddit_data(mock_data)
    print("\nSample cleaned data:")
    print(df_clean[['raw_text', 'clean_text', 'word_count']].to_string())
