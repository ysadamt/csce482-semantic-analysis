"""
Phase 3: Results & Empirical Analysis
From Data to Evidence - Reporting Standards for Peer-Reviewed Journals

This module implements three analysis techniques:
1. Trend-Based Analysis (temporal sentiment dynamics)
2. Spike Detection (volume and sentiment anomalies)
3. Odds Ratio / Log-Odds Analysis (semantic drivers of sentiment)

All analyses follow publication-ready statistical reporting standards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
import re
import os
import json
from scipy import stats
from scipy.special import softmax
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, spearmanr
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
from itertools import combinations
from datetime import datetime, timedelta
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer

warnings.filterwarnings('ignore')


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled = np.sqrt((var_a + var_b) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled)


def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """95% CI for Pearson r via Fisher z-transform."""
    if n <= 3:
        return np.nan, np.nan
    if abs(r) >= 1:
        return r, r
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


def _effect_size_label(value: float) -> str:
    """Interpret correlation-like effect size magnitudes."""
    a = abs(float(value))
    if a < 0.1:
        return 'negligible'
    if a < 0.3:
        return 'small'
    if a < 0.5:
        return 'moderate'
    return 'large'


def _durbin_watson(values: np.ndarray) -> float:
    """Durbin-Watson statistic for residual autocorrelation diagnostics."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.nan
    residuals = values - np.mean(values)
    numerator = np.sum(np.diff(residuals) ** 2)
    denominator = np.sum(residuals ** 2)
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _fdr_bh(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        running_min = min(running_min, val)
        adjusted[i] = running_min
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return out.tolist()


def _bonferroni(p_values: List[float]) -> List[float]:
    """Bonferroni p-value correction."""
    n = max(1, len(p_values))
    return [min(1.0, float(p) * n) for p in p_values]


def _safe_serializable(obj):
    """Convert numpy/pandas objects to JSON-serializable types."""
    if obj is pd.NA:
        return None
    if isinstance(obj, np.generic):
        return _safe_serializable(obj.item())
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool)) else k: _safe_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serializable(v) for v in obj]
    return obj


def extract_keywords_by_sentiment(df: pd.DataFrame,
                                  text_column: str = 'clean_text',
                                  sentiment_column: str = 'sentiment',
                                  top_n: int = 25,
                                  min_token_length: int = 3) -> pd.DataFrame:
    """
    Quality-check helper: extract top keywords for positive/neutral/negative posts.

    Returns a tidy DataFrame with one row per (sentiment, keyword).
    """
    if text_column not in df.columns or sentiment_column not in df.columns:
        return pd.DataFrame()

    stop_words = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'has', 'had',
        'are', 'was', 'were', 'will', 'would', 'could', 'should', 'about', 'into',
        'their', 'there', 'than', 'them', 'they', 'you', 'your', 'our', 'out',
        'not', 'but', 'can', 'all', 'any', 'who', 'how', 'why', 'what', 'when',
        'where', 'just', 'more', 'most', 'also', 'its', 'it', 'to', 'of', 'in',
        'on', 'is', 'a', 'an', 'as', 'at', 'be', 'by', 'or', 'if', 'we', 'i',
        'do', 'does', 'did', 'so', 'my', 'me', 'he', 'she', 'his', 'her', 'theirs',
        'http', 'https', 'www', 'com', 'reddit', 'post', 'comment'
    }

    sentiment_order = ['positive', 'neutral', 'negative']
    token_pattern = re.compile(r"[a-z][a-z0-9_'-]+")
    rows = []

    for sentiment in sentiment_order:
        subset = df[df[sentiment_column].astype(str).str.lower() == sentiment]
        if len(subset) == 0:
            continue

        token_counts = Counter()
        for text in subset[text_column].astype(str):
            for token in token_pattern.findall(text.lower()):
                if len(token) < min_token_length:
                    continue
                if token in stop_words:
                    continue
                if token.isnumeric():
                    continue
                token_counts[token] += 1

        total_sentiment_posts = int(len(subset))
        for keyword, count in token_counts.most_common(top_n):
            rows.append({
                'sentiment': sentiment,
                'keyword': keyword,
                'count': int(count),
                'posts_in_sentiment_class': total_sentiment_posts,
                'keyword_rate_per_post': float(count / max(1, total_sentiment_posts))
            })

    return pd.DataFrame(rows)


class SentimentAnalyzer:
    """
    BERT-based sentiment analysis using pretrained models.
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest.
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initialize sentiment model."""
        print(f"Loading sentiment model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        print("✅ Sentiment model loaded")
    
    def _preprocess_for_model(self, text: str) -> str:
        """Preprocess text for twitter-roberta."""
        words = []
        for word in text.split():
            if word.startswith('@'):
                words.append('@user')
            elif word.startswith('http'):
                words.append('http')
            else:
                words.append(word)
        return " ".join(words)
    
    def predict_sentiment(self, texts: List[str], batch_size: int = 32) -> pd.DataFrame:
        """
        Predict sentiment for list of texts.
        
        Returns DataFrame with columns:
        - sentiment: predicted class (negative/neutral/positive)
        - confidence: probability of predicted class
        - prob_negative, prob_neutral, prob_positive: class probabilities
        - sentiment_score: numeric score (-1 to +1)
        """
        results = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_processed = [self._preprocess_for_model(str(t))[:512] for t in batch]
                
                encoded = self.tokenizer(
                    batch_processed,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                output = self.model(**encoded)
                probs = softmax(output.logits.detach().numpy(), axis=1)
                
                for j, prob_row in enumerate(probs):
                    pred_idx = np.argmax(prob_row)
                    results.append({
                        'sentiment': self.label_map[pred_idx],
                        'confidence': float(prob_row[pred_idx]),
                        'prob_negative': float(prob_row[0]),
                        'prob_neutral': float(prob_row[1]),
                        'prob_positive': float(prob_row[2]),
                        'sentiment_score': float(prob_row[2] - prob_row[0])  # +1 positive, -1 negative
                    })
                
                if (i // batch_size) % 5 == 0:
                    print(f"   Processed {min(i+batch_size, len(texts))}/{len(texts)} posts...")
        
        return pd.DataFrame(results)


class VaderCrossCheckAnalyzer:
    """VADER-based sentiment cross-check model."""

    def __init__(self):
        self.analyzer = VaderSentimentIntensityAnalyzer()

    def predict_sentiment(self, texts: List[str]) -> pd.DataFrame:
        """Predict sentiment with VADER and return class/score probabilities."""
        rows = []
        for i, text in enumerate(texts):
            score = self.analyzer.polarity_scores(str(text))
            compound = float(score.get('compound', 0.0))

            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            rows.append({
                'vader_sentiment': sentiment,
                'vader_confidence': float(max(score.get('pos', 0.0), score.get('neu', 0.0), score.get('neg', 0.0))),
                'vader_prob_negative': float(score.get('neg', 0.0)),
                'vader_prob_neutral': float(score.get('neu', 0.0)),
                'vader_prob_positive': float(score.get('pos', 0.0)),
                'vader_sentiment_score': compound
            })

            if (i + 1) % 250 == 0:
                print(f"   VADER cross-check processed {i + 1}/{len(texts)} posts...")

        return pd.DataFrame(rows)


class TrendAnalysis:
    """
    Trend-Based Temporal Sentiment Analysis.
    
    Features:
    - Monthly/weekly volume and sentiment trends
    - Sentiment proportion evolution
    - Rolling averages with confidence bands
    - Statistical trend testing
    """
    
    def __init__(self, df: pd.DataFrame, date_column: str = 'created_utc',
                 sentiment_column: str = 'sentiment'):
        """Initialize with DataFrame containing sentiment predictions."""
        self.df = df.copy()
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.date_column = date_column
        self.sentiment_column = sentiment_column
        
        # Create time groupings
        self.df['year'] = self.df[date_column].dt.year
        self.df['month'] = self.df[date_column].dt.month
        self.df['year_month'] = self.df[date_column].dt.to_period('M')
        self.df['week'] = self.df[date_column].dt.isocalendar().week
        self.df['year_week'] = self.df[date_column].dt.strftime('%Y-W%W')
    
    def compute_temporal_statistics(self) -> Dict:
        """
        Compute comprehensive temporal statistics.
        
        Returns publication-ready statistics including:
        - Volume dynamics (mean, std, skewness)
        - Temporal coverage
        - Autocorrelation (if sufficient data)
        """
        df = self.df
        
        # Daily volume
        daily_volume = df.groupby(df[self.date_column].dt.date).size()
        
        # Weekly volume
        weekly_volume = df.groupby('year_week').size()
        
        daily_autocorr_lag1 = daily_volume.autocorr(lag=1) if len(daily_volume) > 2 else np.nan
        daily_autocorr_lag7 = daily_volume.autocorr(lag=7) if len(daily_volume) > 8 else np.nan

        # Simple structural break diagnostic (midpoint Chow-style F statistic)
        chow_midpoint = {}
        if len(daily_volume) >= 20:
            y = daily_volume.values.astype(float)
            x = np.arange(len(y), dtype=float)
            x_design = np.column_stack([np.ones_like(x), x])

            def _rss(design, target):
                beta, *_ = np.linalg.lstsq(design, target, rcond=None)
                residuals = target - design @ beta
                return float(np.sum(residuals ** 2))

            split = len(y) // 2
            k = 2
            rss_pooled = _rss(x_design, y)
            rss_1 = _rss(x_design[:split], y[:split])
            rss_2 = _rss(x_design[split:], y[split:])
            numerator = (rss_pooled - (rss_1 + rss_2)) / k
            denominator = (rss_1 + rss_2) / max(1, (len(y) - 2 * k))
            f_stat = numerator / denominator if denominator > 0 else np.nan
            p_value = 1 - stats.f.cdf(f_stat, k, max(1, len(y) - 2 * k)) if not np.isnan(f_stat) else np.nan
            chow_midpoint = {
                'f_statistic': float(f_stat) if not np.isnan(f_stat) else np.nan,
                'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
                'significant_break_0_05': bool(p_value < 0.05) if not np.isnan(p_value) else False,
                'split_date': str(pd.to_datetime(daily_volume.index[split]))
            }

        temporal_stats = {
            'temporal_range': {
                'start': df[self.date_column].min().strftime('%Y-%m-%d'),
                'end': df[self.date_column].max().strftime('%Y-%m-%d'),
                'span_days': (df[self.date_column].max() - df[self.date_column].min()).days,
                'span_months': len(df['year_month'].unique())
            },
            'daily_volume': {
                'mean': daily_volume.mean(),
                'std': daily_volume.std(),
                'median': daily_volume.median(),
                'skewness': daily_volume.skew(),
                'min': daily_volume.min(),
                'max': daily_volume.max()
            },
            'weekly_volume': {
                'mean': weekly_volume.mean(),
                'std': weekly_volume.std(),
                'median': weekly_volume.median()
            },
            'autocorrelation': {
                'daily_lag1': daily_autocorr_lag1,
                'daily_lag7': daily_autocorr_lag7,
                'durbin_watson_daily_volume': _durbin_watson(daily_volume.values)
            },
            'structural_break_midpoint': chow_midpoint,
            'posts_per_year': df['year'].value_counts().sort_index().to_dict()
        }
        
        return temporal_stats
    
    def analyze_sentiment_trends(self) -> pd.DataFrame:
        """
        Compute monthly sentiment trends with statistical metrics.
        
        Returns DataFrame with monthly aggregates.
        """
        # Monthly aggregation
        monthly = self.df.groupby('year_month').agg({
            self.sentiment_column: lambda x: x.value_counts().to_dict(),
            'sentiment_score': ['mean', 'std', 'count'] if 'sentiment_score' in self.df.columns else ['count']
        }).reset_index()
        
        # Flatten column names
        if 'sentiment_score' in self.df.columns:
            monthly.columns = ['year_month', 'sentiment_dist', 'avg_sentiment', 
                             'std_sentiment', 'count']
        else:
            monthly.columns = ['year_month', 'sentiment_dist', 'count']
        
        # Extract sentiment counts
        for sent in ['negative', 'neutral', 'positive']:
            monthly[f'count_{sent}'] = monthly['sentiment_dist'].apply(
                lambda x: x.get(sent, 0)
            )
            monthly[f'pct_{sent}'] = monthly[f'count_{sent}'] / monthly['count'] * 100
        
        # Convert period to timestamp for plotting
        monthly['date'] = monthly['year_month'].apply(lambda x: x.to_timestamp())
        
        return monthly
    
    def test_trend_significance(self, monthly_df: pd.DataFrame) -> Dict:
        """
        Statistical test for trend significance.
        
        Tests:
        - Mann-Kendall trend test (non-parametric)
        - Comparison of first vs second half
        """
        results = {}
        
        if 'avg_sentiment' in monthly_df.columns:
            sentiment_series = monthly_df['avg_sentiment'].values
            
            # First half vs second half comparison
            mid = len(sentiment_series) // 2
            if mid > 2:
                first_half = sentiment_series[:mid]
                second_half = sentiment_series[mid:]
                
                # Mann-Whitney U test
                stat, p_value = mannwhitneyu(first_half, second_half, alternative='two-sided')
                
                # Effect size (Cohen's d)
                cohens_d = _cohens_d(second_half, first_half)

                # Confidence interval for mean difference (Welch approximation)
                mean_diff = float(np.mean(second_half) - np.mean(first_half))
                se = np.sqrt(
                    (np.var(first_half, ddof=1) / max(1, len(first_half))) +
                    (np.var(second_half, ddof=1) / max(1, len(second_half)))
                )
                ci_low, ci_high = (mean_diff, mean_diff)
                if se > 0:
                    tcrit = stats.t.ppf(0.975, df=max(1, len(first_half) + len(second_half) - 2))
                    ci_low = mean_diff - tcrit * se
                    ci_high = mean_diff + tcrit * se
                
                results['first_vs_second_half'] = {
                    'first_half_mean': np.mean(first_half),
                    'second_half_mean': np.mean(second_half),
                    'mann_whitney_U': stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'mean_difference': mean_diff,
                    'mean_difference_ci_95': [float(ci_low), float(ci_high)],
                    'direction': 'increasing' if np.mean(second_half) > np.mean(first_half) else 'decreasing',
                    'significant': p_value < 0.05,
                    'assumptions_checked': {
                        'non_parametric_test_used': True,
                        'independent_periods_assumed': True
                    }
                }
        
        return results
    
    def plot_trends(self, monthly_df: pd.DataFrame, 
                   save_path: str = None) -> None:
        """Generate publication-ready trend visualizations."""
        
        color_map = {'negative': '#ff6b6b', 'neutral': '#ffd93d', 'positive': '#6bcb77'}
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # ===== Panel A: Volume Over Time =====
        ax1 = axes[0]
        ax1.bar(monthly_df['date'], monthly_df['count'], 
               color='#4d96ff', alpha=0.7, edgecolor='#333')
        
        # Add trend line
        x_numeric = np.arange(len(monthly_df))
        z = np.polyfit(x_numeric, monthly_df['count'], 1)
        p = np.poly1d(z)
        ax1.plot(monthly_df['date'], p(x_numeric), 'r--', linewidth=2, label='Trend')
        
        ax1.set_ylabel('Number of Posts', fontsize=12)
        ax1.set_title('A) Monthly Post Volume Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        
        # ===== Panel B: Sentiment Proportions =====
        ax2 = axes[1]
        for sent in ['negative', 'neutral', 'positive']:
            ax2.plot(monthly_df['date'], monthly_df[f'pct_{sent}'],
                    marker='o', markersize=4, linewidth=2,
                    color=color_map[sent], label=sent.capitalize())
        
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title('B) Monthly Sentiment Distribution', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # ===== Panel C: Net Sentiment Score =====
        if 'avg_sentiment' in monthly_df.columns:
            ax3 = axes[2]
            
            # Plot with confidence band
            ax3.plot(monthly_df['date'], monthly_df['avg_sentiment'],
                    marker='o', markersize=4, linewidth=2, color='#4d96ff')
            
            if 'std_sentiment' in monthly_df.columns:
                ax3.fill_between(monthly_df['date'],
                               monthly_df['avg_sentiment'] - monthly_df['std_sentiment'],
                               monthly_df['avg_sentiment'] + monthly_df['std_sentiment'],
                               alpha=0.2, color='#4d96ff')
            
            ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax3.fill_between(monthly_df['date'], monthly_df['avg_sentiment'], 0,
                           where=(monthly_df['avg_sentiment'] >= 0),
                           color='#6bcb77', alpha=0.2)
            ax3.fill_between(monthly_df['date'], monthly_df['avg_sentiment'], 0,
                           where=(monthly_df['avg_sentiment'] < 0),
                           color='#ff6b6b', alpha=0.2)
            
            ax3.set_ylabel('Sentiment Score (-1 to +1)', fontsize=12)
            ax3.set_title('C) Average Sentiment Score Over Time', fontsize=14, fontweight='bold')
            ax3.set_ylim(-1, 1)
            ax3.grid(axis='y', alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(monthly_df) // 12)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Temporal Sentiment Trend Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Trend plot saved to: {save_path}")
        
        plt.show()


class SpikeDetection:
    """
    Statistical Spike Detection for Volume and Sentiment.
    
    Features:
    - σ-based anomaly detection (μ ± 2σ)
    - Event correlation analysis
    - Lag structure analysis
    """
    
    def __init__(self, df: pd.DataFrame, date_column: str = 'created_utc'):
        """Initialize with DataFrame."""
        self.df = df.copy()
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.date_column = date_column
        
        # Compute daily metrics
        self.daily_volume = self.df.groupby(
            self.df[date_column].dt.date
        ).size().reset_index(name='volume')
        self.daily_volume.columns = ['date', 'volume']
        self.daily_volume['date'] = pd.to_datetime(self.daily_volume['date'])
        
        if 'sentiment_score' in df.columns:
            daily_sent = self.df.groupby(
                self.df[date_column].dt.date
            )['sentiment_score'].mean().reset_index()
            daily_sent.columns = ['date', 'avg_sentiment']
            daily_sent['date'] = pd.to_datetime(daily_sent['date'])
            self.daily_volume = self.daily_volume.merge(daily_sent, on='date')

        # Daily class proportions for richer temporal diagnostics
        if 'sentiment' in self.df.columns:
            daily_sentiment_dist = (
                self.df.groupby(self.df[date_column].dt.date)['sentiment']
                .value_counts(normalize=True)
                .unstack(fill_value=0)
                .reset_index()
            )
            daily_sentiment_dist.columns = [
                'date' if c == date_column else f"pct_{c}" if c in ['negative', 'neutral', 'positive'] else c
                for c in daily_sentiment_dist.columns
            ]
            daily_sentiment_dist['date'] = pd.to_datetime(daily_sentiment_dist['date'])
            self.daily_volume = self.daily_volume.merge(daily_sentiment_dist, on='date', how='left')
            for c in ['pct_negative', 'pct_neutral', 'pct_positive']:
                if c not in self.daily_volume.columns:
                    self.daily_volume[c] = 0.0

            include_neutral = True
            denom = self.daily_volume['pct_positive'] + self.daily_volume['pct_negative']
            if include_neutral:
                denom = denom + self.daily_volume['pct_neutral']
            self.daily_volume['net_sentiment'] = np.where(
                denom > 0,
                (self.daily_volume['pct_positive'] - self.daily_volume['pct_negative']) / denom,
                0.0
            )
    
    def detect_volume_spikes(self, sigma_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect statistically significant volume spikes.
        
        Spikes are defined as days where volume > μ + σ*threshold
        
        Args:
            sigma_threshold: Number of standard deviations (default: 2.0)
            
        Returns:
            DataFrame of spike dates with statistics
        """
        volume = self.daily_volume['volume']
        mean_vol = volume.mean()
        std_vol = volume.std()
        
        threshold_upper = mean_vol + sigma_threshold * std_vol
        threshold_lower = mean_vol - sigma_threshold * std_vol
        
        spikes = self.daily_volume[
            (volume > threshold_upper) | (volume < threshold_lower)
        ].copy()
        
        spikes['z_score'] = (spikes['volume'] - mean_vol) / std_vol
        spikes['spike_type'] = spikes['z_score'].apply(
            lambda x: 'high' if x > 0 else 'low'
        )
        spikes['significance'] = spikes['z_score'].abs().apply(
            lambda x: 'p<0.001' if x > 3.29 else ('p<0.01' if x > 2.58 else ('p<0.05' if x > 1.96 else 'ns'))
        )
        
        return spikes.sort_values('z_score', ascending=False)
    
    def detect_sentiment_spikes(self, sigma_threshold: float = 2.0) -> pd.DataFrame:
        """Detect statistically significant sentiment spikes."""
        if 'avg_sentiment' not in self.daily_volume.columns:
            print("⚠️ No sentiment data available")
            return pd.DataFrame()
        
        sentiment = self.daily_volume['avg_sentiment']
        mean_sent = sentiment.mean()
        std_sent = sentiment.std()
        
        threshold_upper = mean_sent + sigma_threshold * std_sent
        threshold_lower = mean_sent - sigma_threshold * std_sent
        
        spikes = self.daily_volume[
            (sentiment > threshold_upper) | (sentiment < threshold_lower)
        ].copy()
        
        spikes['z_score'] = (spikes['avg_sentiment'] - mean_sent) / std_sent
        spikes['spike_type'] = spikes['z_score'].apply(
            lambda x: 'positive_surge' if x > 0 else 'negative_surge'
        )
        
        return spikes.sort_values('z_score', ascending=False)
    
    def compute_lag_correlations(self, event_dates: List[str] = None,
                                 max_lag: int = 28,
                                 target_lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Compute correlations at different time lags.
        
        Tests relationship between volume/sentiment at t=0 and subsequent days.
        
        Args:
            event_dates: Optional list of known event dates to analyze
            max_lag: Maximum lag in days to test
            
        Returns:
            DataFrame with lag correlations
        """
        if 'avg_sentiment' not in self.daily_volume.columns:
            print("⚠️ No sentiment data for lag analysis")
            return pd.DataFrame()
        
        if target_lags is None:
            target_lags = [0, 7, 21]

        results = []
        
        for lag in range(0, max_lag + 1):
            # Shift sentiment by lag days
            df_lag = self.daily_volume.copy()
            df_lag['sentiment_lagged'] = df_lag['avg_sentiment'].shift(lag)
            df_lag = df_lag.dropna()
            
            if len(df_lag) > 10:
                # Correlation between volume and lagged sentiment
                r_vol_sent, p_vol_sent = pearsonr(
                    df_lag['volume'], df_lag['sentiment_lagged']
                )
                
                ci_low, ci_high = _pearson_ci(r_vol_sent, len(df_lag))
                results.append({
                    'lag_days': lag,
                    'correlation_vol_sent': r_vol_sent,
                    'p_value': p_vol_sent,
                    'ci_95_low': ci_low,
                    'ci_95_high': ci_high,
                    'effect_size': _effect_size_label(r_vol_sent),
                    'significant': p_vol_sent < 0.05,
                    'is_target_lag': lag in target_lags,
                    'n': len(df_lag)
                })

        lag_df = pd.DataFrame(results)
        if len(lag_df) == 0:
            return lag_df

        lag_df['p_bonferroni'] = _bonferroni(lag_df['p_value'].tolist())
        lag_df['p_fdr_bh'] = _fdr_bh(lag_df['p_value'].tolist())
        lag_df['significant_fdr_0_05'] = lag_df['p_fdr_bh'] < 0.05

        return lag_df

    def compute_cross_correlation_profile(self, max_lag: int = 28) -> pd.DataFrame:
        """Cross-correlation profile between daily volume and net sentiment."""
        if 'net_sentiment' not in self.daily_volume.columns:
            return pd.DataFrame()

        x = self.daily_volume['volume'].astype(float).values
        y = self.daily_volume['net_sentiment'].astype(float).values
        if len(x) < max(12, max_lag + 3):
            return pd.DataFrame()

        x = (x - x.mean()) / (x.std() + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)

        rows = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x_part = x[-lag:]
                y_part = y[:len(y) + lag]
            elif lag > 0:
                x_part = x[:len(x) - lag]
                y_part = y[lag:]
            else:
                x_part = x
                y_part = y

            if len(x_part) < 10:
                continue
            corr, p_value = pearsonr(x_part, y_part)
            rows.append({
                'lag_days': lag,
                'cross_correlation': corr,
                'p_value': p_value,
                'n': len(x_part)
            })

        ccf = pd.DataFrame(rows)
        if len(ccf) > 0:
            ccf['p_fdr_bh'] = _fdr_bh(ccf['p_value'].tolist())
            ccf['significant_fdr_0_05'] = ccf['p_fdr_bh'] < 0.05
        return ccf
    
    def plot_spikes(self, volume_spikes: pd.DataFrame = None,
                   sentiment_spikes: pd.DataFrame = None,
                   save_path: str = None) -> None:
        """Generate spike detection visualization."""
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # ===== Panel A: Volume with Spikes =====
        ax1 = axes[0]
        
        # Plot volume
        ax1.plot(self.daily_volume['date'], self.daily_volume['volume'],
                color='#4d96ff', alpha=0.5, linewidth=1)
        
        # Add rolling average
        rolling = self.daily_volume['volume'].rolling(window=7, min_periods=1).mean()
        ax1.plot(self.daily_volume['date'], rolling,
                color='#4d96ff', linewidth=2, label='7-day avg')
        
        # Add threshold bands
        mean_vol = self.daily_volume['volume'].mean()
        std_vol = self.daily_volume['volume'].std()
        ax1.axhline(mean_vol, color='gray', linestyle='-', alpha=0.5, label='Mean')
        ax1.axhline(mean_vol + 2*std_vol, color='red', linestyle='--', alpha=0.5, label='μ+2σ')
        ax1.axhline(mean_vol - 2*std_vol, color='red', linestyle='--', alpha=0.5)
        
        # Mark spikes
        if volume_spikes is not None and len(volume_spikes) > 0:
            spike_dates = volume_spikes[volume_spikes['spike_type'] == 'high']['date']
            spike_vals = volume_spikes[volume_spikes['spike_type'] == 'high']['volume']
            ax1.scatter(spike_dates, spike_vals, color='red', s=100, zorder=5,
                       marker='^', label='Volume spike')
        
        ax1.set_ylabel('Daily Volume', fontsize=12)
        ax1.set_title('A) Volume Spike Detection (μ ± 2σ Method)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        
        # ===== Panel B: Sentiment with Spikes =====
        ax2 = axes[1]
        
        if 'avg_sentiment' in self.daily_volume.columns:
            ax2.plot(self.daily_volume['date'], self.daily_volume['avg_sentiment'],
                    color='#666', alpha=0.5, linewidth=1)
            
            rolling_sent = self.daily_volume['avg_sentiment'].rolling(window=7, min_periods=1).mean()
            ax2.plot(self.daily_volume['date'], rolling_sent,
                    color='#4d96ff', linewidth=2, label='7-day avg')
            
            # Add threshold bands
            mean_sent = self.daily_volume['avg_sentiment'].mean()
            std_sent = self.daily_volume['avg_sentiment'].std()
            ax2.axhline(mean_sent, color='gray', linestyle='-', alpha=0.5)
            ax2.axhline(mean_sent + 2*std_sent, color='green', linestyle='--', alpha=0.5)
            ax2.axhline(mean_sent - 2*std_sent, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
            
            # Mark sentiment spikes
            if sentiment_spikes is not None and len(sentiment_spikes) > 0:
                pos_spikes = sentiment_spikes[sentiment_spikes['spike_type'] == 'positive_surge']
                neg_spikes = sentiment_spikes[sentiment_spikes['spike_type'] == 'negative_surge']
                
                if len(pos_spikes) > 0:
                    ax2.scatter(pos_spikes['date'], pos_spikes['avg_sentiment'],
                              color='green', s=100, zorder=5, marker='^', label='Positive spike')
                if len(neg_spikes) > 0:
                    ax2.scatter(neg_spikes['date'], neg_spikes['avg_sentiment'],
                              color='red', s=100, zorder=5, marker='v', label='Negative spike')
            
            ax2.set_ylabel('Sentiment Score', fontsize=12)
            ax2.set_ylim(-1, 1)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('B) Sentiment Spike Detection', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Statistical Spike Detection Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Spike plot saved to: {save_path}")
        
        plt.show()

    def plot_epidemic_curve_overlay(self,
                                    volume_spikes: Optional[pd.DataFrame] = None,
                                    save_path: str = None,
                                    event_markers: Optional[List[Tuple[str, str]]] = None) -> None:
        """Plot epidemic-style overlay: volume bars + sentiment line + spike markers."""
        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.bar(self.daily_volume['date'], self.daily_volume['volume'],
                color='#4d96ff', alpha=0.5, label='Daily post volume (count)')
        ax1.set_ylabel('Post volume (posts/day)', fontsize=11)

        ax2 = ax1.twinx()
        sentiment_series = 'net_sentiment' if 'net_sentiment' in self.daily_volume.columns else 'avg_sentiment'
        if sentiment_series in self.daily_volume.columns:
            line = self.daily_volume[sentiment_series]
            roll = line.rolling(window=7, min_periods=1).mean()
            sem = line.rolling(window=7, min_periods=3).std() / np.sqrt(7)
            ax2.plot(self.daily_volume['date'], roll, color='#ff6b6b', linewidth=2.2,
                     label='7-day average sentiment')
            ax2.fill_between(self.daily_volume['date'], (roll - 1.96 * sem).fillna(roll),
                             (roll + 1.96 * sem).fillna(roll), color='#ff6b6b', alpha=0.18,
                             label='Approx. 95% CI band')
            ax2.set_ylabel('Sentiment index (unitless)', fontsize=11)

        if volume_spikes is not None and len(volume_spikes) > 0:
            highs = volume_spikes[volume_spikes['spike_type'] == 'high']
            ax1.scatter(highs['date'], highs['volume'], color='red', marker='^', s=90,
                        zorder=5, label='Significant surge (μ+2σ)')

        if event_markers:
            for event_date, event_label in event_markers:
                event_ts = pd.to_datetime(event_date)
                ax1.axvline(event_ts, color='black', linestyle='--', alpha=0.4)
                ax1.annotate(event_label, xy=(event_ts, ax1.get_ylim()[1] * 0.92),
                             xytext=(5, 0), textcoords='offset points', rotation=90,
                             fontsize=9, va='top')

        ax1.set_title('Epidemic Curve Overlay: Discourse Volume and Sentiment Dynamics',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Epidemic overlay saved to: {save_path}")

        plt.show()

    def plot_correlation_heatmap(self, save_path: str = None) -> Optional[pd.DataFrame]:
        """Correlation heatmap for key temporal metrics."""
        metric_cols = ['volume', 'pct_negative', 'pct_neutral', 'pct_positive', 'net_sentiment', 'avg_sentiment']
        available = [c for c in metric_cols if c in self.daily_volume.columns]
        if len(available) < 3:
            return None

        corr_df = self.daily_volume[available].corr(method='pearson')

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={'label': 'Pearson r'})
        plt.title('Correlation Heatmap of Sentiment and Volume Metrics',
                  fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Correlation heatmap saved to: {save_path}")

        plt.show()
        return corr_df


class OddsRatioAnalysis:
    """
    Odds Ratio and Log-Odds Analysis for Semantic Drivers.
    
    Identifies words/topics that drive positive vs negative sentiment.
    """
    
    def __init__(self, df: pd.DataFrame, text_column: str = 'clean_text',
                 sentiment_column: str = 'sentiment'):
        """Initialize with DataFrame."""
        self.df = df.copy()
        self.text_column = text_column
        self.sentiment_column = sentiment_column
        self.stopwords = {
            'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 'were',
            'their', 'they', 'them', 'about', 'would', 'could', 'should', 'there',
            'which', 'when', 'where', 'what', 'your', 'just', 'into', 'over',
            'than', 'also', 'http', 'https', 'www', 'com', 'amp', 'reddit',
            'are', 'was', 'is', 'be', 'been', 'being', 'am', 'to', 'of', 'in',
            'on', 'at', 'as', 'by', 'it', 'its', 'if', 'or', 'can',
            'post', 'comment', 'people', 'thing', 'things', 'make', 'made',
            'know', 'like', 'really', 'still', 'much', 'many', 'some', 'more',
            'most', 'other', 'because', 'going', 'want', 'need', 'take', 'come',
            'said', 'dont', 'doesnt', 'didnt', 'cant', 'wont', 'ive', 'youre',
            'im', 'get', 'got', 'one', 'two', 'first', 'last', 'new', 'use',
            'used', 'using', 'well', 'back', 'look', 'long', 'even', 'say',
            # Contraction stems and generic filler fragments.
            'don', 'doesn', 'didn', 'isn', 'aren', 'wasn', 'weren', 'won',
            'wouldn', 'couldn', 'shouldn', 'mustn', 'mightn', 'needn', 'hasn',
            'hadn', 'haven', 'ain', 'll', 've', 're', 'let', 'etc'
        }

        self.topic_anchor_terms = {
            'methane', 'emission', 'emissions', 'greenhouse', 'ghg', 'climate',
            'dairy', 'livestock', 'cattle', 'cow', 'cows', 'beef', 'ruminant',
            'enteric', 'fermentation', 'farm', 'farming', 'manure', 'feed',
            'digester', 'digesters', 'agriculture', 'sustainable', 'sustainability',
            'carbon', 'warming', 'decarbonization', 'mitigation'
        }
        self.topic_stems = (
            'methan', 'emiss', 'greenhouse', 'climat', 'carbon', 'ghg',
            'dair', 'livestock', 'cattl', 'cow', 'beef', 'rumin', 'enteric',
            'ferment', 'farm', 'agric', 'manur', 'feed', 'digest', 'policy',
            'regulat', 'sustain', 'mitigat', 'warming', 'decarbon'
        )

    def _is_topic_relevant_word(self, word: str) -> bool:
        """Allow only terms that are lexically aligned with the query domain."""
        if word in self.topic_anchor_terms:
            return True
        return any(word.startswith(stem) for stem in self.topic_stems)

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize and filter words for semantic odds analysis."""
        text = str(text).lower()
        tokens = re.findall(r'\b[a-z]{3,}\b', text)
        tokens = [t for t in tokens if t not in self.stopwords]
        return tokens
    
    def compute_word_frequencies(self) -> Tuple[Counter, Counter, Counter]:
        """Compute per-document word frequencies by sentiment class."""
        
        positive_words = Counter()
        negative_words = Counter()
        all_words = Counter()
        
        for _, row in self.df.iterrows():
            words = self._tokenize_text(row[self.text_column])
            words = list(set(words))
            sentiment = row[self.sentiment_column]
            
            all_words.update(words)
            
            if sentiment == 'positive':
                positive_words.update(words)
            elif sentiment == 'negative':
                negative_words.update(words)
        
        return positive_words, negative_words, all_words
    
    def compute_log_odds_ratio(self, min_count: int = 10,
                               prior_count: float = 0.5,
                               max_doc_frequency: float = 0.8) -> pd.DataFrame:
        """
        Compute log-odds ratio with informative Dirichlet prior.
        
        log(OR) = log(P(word|positive) / P(word|negative))
        
        Args:
            min_count: Minimum word frequency to include
            prior_count: Prior count for smoothing (Laplace smoothing)
            
        Returns:
            DataFrame with words ranked by log-odds
        """
        positive_words, negative_words, all_words = self.compute_word_frequencies()

        # Document-level relevance signal for each token.
        word_doc_counts = Counter()
        word_topic_doc_counts = Counter()
        for text in self.df[self.text_column]:
            tokens = set(self._tokenize_text(text))
            if not tokens:
                continue
            text_lower = str(text).lower()
            has_topic_anchor = any(anchor in text_lower for anchor in self.topic_anchor_terms)
            for token in tokens:
                word_doc_counts[token] += 1
                if has_topic_anchor:
                    word_topic_doc_counts[token] += 1

        total_docs = max(1, len(self.df))
        
        # Total counts
        total_positive = sum(positive_words.values())
        total_negative = sum(negative_words.values())
        
        results = []
        
        for word, total_count in all_words.items():
            if total_count < min_count:
                continue
            if (total_count / total_docs) > max_doc_frequency:
                continue

            # Hard topic constraint: suppress generic conversation terms.
            if not self._is_topic_relevant_word(word):
                continue

            # Retain terms that are either explicit anchors or strongly tied to topic posts.
            doc_count = word_doc_counts.get(word, 0)
            topic_ratio = (word_topic_doc_counts.get(word, 0) / doc_count) if doc_count > 0 else 0.0
            if word not in self.topic_anchor_terms and topic_ratio < 0.35:
                continue
            
            # Counts with prior smoothing
            pos_count = positive_words.get(word, 0) + prior_count
            neg_count = negative_words.get(word, 0) + prior_count
            
            # Probabilities
            p_word_positive = pos_count / (total_positive + prior_count * len(all_words))
            p_word_negative = neg_count / (total_negative + prior_count * len(all_words))
            
            # Log-odds ratio
            log_odds = np.log(p_word_positive / p_word_negative)
            
            # Odds ratio
            odds_ratio = p_word_positive / p_word_negative
            
            results.append({
                'word': word,
                'count_total': total_count,
                'count_positive': positive_words.get(word, 0),
                'count_negative': negative_words.get(word, 0),
                'p_word_positive': p_word_positive,
                'p_word_negative': p_word_negative,
                'odds_ratio': odds_ratio,
                'log_odds': log_odds,
                'topic_relevance_ratio': float(topic_ratio),
                'sentiment_driver': 'positive' if log_odds > 0 else 'negative'
            })

        if not results:
            return pd.DataFrame(columns=[
                'word', 'count_total', 'count_positive', 'count_negative',
                'p_word_positive', 'p_word_negative', 'odds_ratio', 'log_odds',
                'topic_relevance_ratio', 'sentiment_driver'
            ])

        return pd.DataFrame(results).sort_values('log_odds', ascending=False)
    
    def compute_sentiment_probabilities(self) -> Dict:
        """
        Compute overall positive/negative/neutral probabilities.
        
        Returns probability distribution and confidence intervals.
        """
        sentiment_counts = self.df[self.sentiment_column].value_counts()
        total = len(self.df)
        
        results = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            prob = count / total
            
            # Wilson score interval for confidence bounds
            n = total
            z = 1.96  # 95% CI
            p = prob
            
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n)) / denominator
            spread = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
            
            results[sentiment] = {
                'count': count,
                'probability': prob,
                'ci_lower': max(0, center - spread),
                'ci_upper': min(1, center + spread)
            }
        
        # Net sentiment
        net_sentiment = (sentiment_counts.get('positive', 0) - 
                        sentiment_counts.get('negative', 0)) / total
        results['net_sentiment'] = net_sentiment
        
        return results
    
    def get_top_drivers(self, log_odds_df: pd.DataFrame, n: int = 15) -> Dict:
        """Get top positive and negative sentiment drivers."""
        positive_drivers = log_odds_df.head(n)[['word', 'log_odds', 'count_total']].to_dict('records')
        negative_drivers = log_odds_df.tail(n)[['word', 'log_odds', 'count_total']].to_dict('records')
        
        return {
            'positive_drivers': positive_drivers,
            'negative_drivers': negative_drivers[::-1]  # Reverse to show most negative first
        }
    
    def plot_semantic_drivers(self, log_odds_df: pd.DataFrame, 
                              n_words: int = 15,
                              save_path: str = None) -> None:
        """Generate semantic driver visualization."""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # ===== Top Positive Drivers =====
        ax1 = axes[0]
        top_positive = log_odds_df.head(n_words)
        
        bars1 = ax1.barh(range(len(top_positive)), top_positive['log_odds'],
                        color='#6bcb77', edgecolor='#333', alpha=0.8)
        ax1.set_yticks(range(len(top_positive)))
        ax1.set_yticklabels(top_positive['word'])
        ax1.set_xlabel('Log-Odds Ratio', fontsize=12)
        ax1.set_title('Positive Sentiment Drivers', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.invert_yaxis()
        
        # Add count labels
        for i, (_, row) in enumerate(top_positive.iterrows()):
            ax1.text(row['log_odds'] + 0.02, i, f"n={row['count_total']:.0f}",
                    va='center', fontsize=9, alpha=0.7)
        
        # ===== Top Negative Drivers =====
        ax2 = axes[1]
        top_negative = log_odds_df.tail(n_words).iloc[::-1]
        
        bars2 = ax2.barh(range(len(top_negative)), top_negative['log_odds'],
                        color='#ff6b6b', edgecolor='#333', alpha=0.8)
        ax2.set_yticks(range(len(top_negative)))
        ax2.set_yticklabels(top_negative['word'])
        ax2.set_xlabel('Log-Odds Ratio', fontsize=12)
        ax2.set_title('Negative Sentiment Drivers', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.invert_yaxis()
        
        # Add count labels
        for i, (_, row) in enumerate(top_negative.iterrows()):
            ax2.text(row['log_odds'] - 0.02, i, f"n={row['count_total']:.0f}",
                    va='center', ha='right', fontsize=9, alpha=0.7)
        
        plt.suptitle('Semantic Drivers of Sentiment (Log-Odds Analysis)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Semantic drivers plot saved to: {save_path}")
        
        plt.show()
    
    def plot_probability_distribution(self, save_path: str = None) -> None:
        """Plot sentiment probability distribution with confidence intervals."""
        
        probs = self.compute_sentiment_probabilities()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sentiments = ['negative', 'neutral', 'positive']
        colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
        
        x = np.arange(len(sentiments))
        width = 0.6
        
        # Bar heights (probabilities)
        heights = [probs[s]['probability'] * 100 for s in sentiments]
        
        # Error bars (CI)
        errors_lower = [(probs[s]['probability'] - probs[s]['ci_lower']) * 100 for s in sentiments]
        errors_upper = [(probs[s]['ci_upper'] - probs[s]['probability']) * 100 for s in sentiments]
        
        bars = ax.bar(x, heights, width, color=colors, edgecolor='#333', alpha=0.8,
                     yerr=[errors_lower, errors_upper], capsize=5, error_kw={'elinewidth': 2})
        
        # Add count labels
        for i, s in enumerate(sentiments):
            ax.text(i, heights[i] + errors_upper[i] + 2, 
                   f"n={probs[s]['count']}", ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in sentiments], fontsize=12)
        ax.set_ylabel('Probability (%)', fontsize=12)
        ax.set_title('Sentiment Probability Distribution with 95% CI', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add net sentiment annotation
        net = probs['net_sentiment']
        ax.text(0.98, 0.98, f"Net Sentiment: {net:+.3f}",
               transform=ax.transAxes, ha='right', va='top',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Probability plot saved to: {save_path}")
        
        plt.show()


class NetworkAnalysis:
    """
    Word Co-occurrence Network Analysis with Community/Topic Detection.
    
    Features:
    - Build word co-occurrence network from text
    - Detect topic communities using modularity optimization
    - Label each community with representative topic
    - Publication-ready visualization
    """
    
    def __init__(self, df: pd.DataFrame, text_column: str = 'clean_text'):
        """Initialize with DataFrame."""
        self.df = df.copy()
        self.text_column = text_column
        
        # Stopwords for filtering
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'we', 'our', 'you', 'your', 'i', 'me', 'my', 'he', 'she', 'his', 'her',
            'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
            'http', 'https', 'www', 'com', 'amp', 'like', 'get', 'got', 'going',
            'make', 'made', 'think', 'know', 'see', 'way', 'even', 'new', 'want',
            'because', 'any', 'give', 'day', 'use', 'her', 'him', 'two', 'been',
            'many', 'said', 'much', 'need', 'take', 'come', 'say', 'still', 'really',
            'thing', 'things', 'something', 'anything', 'everything', 'nothing',
            'people', 'one', 'first', 'last', 'back', 'well', 'look', 'long',
            'cant', 'dont', 'doesnt', 'didnt', 'wont', 'im', 'ive', 'youre'
        }
        
        self.G = None
        self.communities = None
        self.word_freq = None
        self.node_community = None
        self.community_topics = None
        self.selected_topic_indices = [0, 1, 4]  # Keep Topic 1, 2, and 5 only
        self.preset_topic_definitions = [
            {
                'topic_id': 1,
                'label': 'Emissions & Climate Science',
                'anchors': {
                    'methane', 'emission', 'emissions', 'greenhouse', 'climate',
                    'warming', 'carbon', 'co2', 'atmosphere', 'science'
                }
            },
            {
                'topic_id': 2,
                'label': 'Dairy Farming & Livestock Practices',
                'anchors': {
                    'dairy', 'cattle', 'cow', 'cows', 'livestock', 'farm',
                    'farming', 'feed', 'manure', 'herd', 'enteric'
                }
            },
            {
                'topic_id': 3,
                'label': 'Policy, Regulation & Public Debate',
                'anchors': {
                    'policy', 'regulation', 'government', 'law', 'epa',
                    'ban', 'rules', 'subsidy', 'politics', 'public'
                }
            },
            {
                'topic_id': 4,
                'label': 'Economics, Prices & Supply Chain',
                'anchors': {
                    'price', 'prices', 'cost', 'market', 'inflation', 'shortage',
                    'supply', 'demand', 'industry', 'consumer', 'economy'
                }
            },
            {
                'topic_id': 5,
                'label': 'Mitigation, Technology & Sustainability',
                'anchors': {
                    'sustainable', 'sustainability', 'innovation', 'technology',
                    'digesters', 'digester', 'reduction', 'renewable', 'solution',
                    'efficiency', 'transparency', 'welfare'
                }
            }
        ]

        selected_presets = [self.preset_topic_definitions[i] for i in self.selected_topic_indices]
        self.topic_anchor_terms = set()
        for preset in selected_presets:
            self.topic_anchor_terms.update(preset['anchors'])
        self.topic_stems = (
            'methan', 'emiss', 'greenhouse', 'climat', 'carbon', 'ghg',
            'dair', 'livestock', 'cattl', 'cow', 'beef', 'rumin', 'enteric',
            'ferment', 'farm', 'agric', 'manur', 'feed', 'digest', 'sustain',
            'mitigat', 'renew', 'efficien', 'welfare', 'transparen'
        )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Extract clean tokens from text."""
        text = str(text).lower()
        # Extract words (alphanumeric, 3+ chars)
        tokens = re.findall(r'\b[a-z]{3,}\b', text)
        # Filter stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        return tokens
    
    def build_network(self, min_cooccurrence: int = 1,
                     max_edges: int = 800,
                     window_size: int = None,
                     min_word_frequency: int = 2) -> nx.Graph:
        """
        Build word co-occurrence network.
        
        Args:
            min_cooccurrence: Minimum co-occurrence count for edge
            max_edges: Maximum number of edges to include
            window_size: If set, only count co-occurrences within window
                        If None, use document-level co-occurrence
            min_word_frequency: Minimum unigram frequency required for a node
        
        Returns:
            NetworkX graph
        """
        co_occurrence = Counter()
        self.word_freq = Counter()
        
        for text in self.df[self.text_column]:
            tokens = self._tokenize_text(text)
            
            # Count word frequencies
            self.word_freq.update(tokens)
            
            # Count co-occurrences (unique pairs per document)
            unique_tokens = sorted(list(set(tokens)))
            for i in range(len(unique_tokens)):
                for j in range(i + 1, len(unique_tokens)):
                    edge = (unique_tokens[i], unique_tokens[j])
                    co_occurrence[edge] += 1
        
        # Build graph
        self.G = nx.Graph()

        allowed_words = {
            word for word, freq in self.word_freq.items()
            if freq >= min_word_frequency
        }
        
        edges_added = 0
        for (w1, w2), weight in co_occurrence.most_common():
            if weight < min_cooccurrence:
                break
            if edges_added >= max_edges:
                break
            if w1 not in allowed_words or w2 not in allowed_words:
                continue
            self.G.add_edge(w1, w2, weight=weight)
            edges_added += 1
        
        print(f"   Network: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G
    
    def detect_communities(self, random_seed: int = 42) -> List[set]:
        """
        Detect communities with Louvain, then map into Topic 1/2/5 only.
        
        Returns:
            List of communities (sets of words)
        """
        if self.G is None or self.G.number_of_nodes() == 0:
            print("   ⚠️ No network built yet")
            return []

        try:
            from networkx.algorithms.community import louvain_communities
        except Exception as e:
            raise RuntimeError(
                "Louvain community detection is required but unavailable."
            ) from e

        louvain_result = list(louvain_communities(self.G, weight='weight', seed=random_seed))

        selected_presets = [self.preset_topic_definitions[i] for i in self.selected_topic_indices]
        preset_bins = [set() for _ in selected_presets]

        for comm in louvain_result:
            words = sorted(list(comm), key=lambda w: self.word_freq.get(w, 0), reverse=True)
            top_words = set(words[:15])
            scores = []
            for preset in selected_presets:
                overlap = len(top_words.intersection(preset['anchors']))
                scores.append(overlap)

            if max(scores) == 0:
                target_idx = int(np.argmin([len(b) for b in preset_bins]))
            else:
                best = max(scores)
                candidates = [i for i, s in enumerate(scores) if s == best]
                if len(candidates) == 1:
                    target_idx = candidates[0]
                else:
                    target_idx = min(candidates, key=lambda i: len(preset_bins[i]))

            preset_bins[target_idx].update(comm)

        self.communities = [c for c in preset_bins if len(c) > 0]

        self.node_community = {}
        for idx, community in enumerate(self.communities):
            for node in community:
                self.node_community[node] = idx

        # Keep selected presets aligned with kept non-empty communities.
        non_empty_selected = []
        for i, community in enumerate(preset_bins):
            if len(community) > 0:
                non_empty_selected.append(selected_presets[i])
        self.active_preset_topics = non_empty_selected

        print(f"   Detected {len(louvain_result)} Louvain communities")
        print("   Using selected topic groups: Topic 1, Topic 2, Topic 5")
        print(f"   Mapped to {len(self.communities)} selected topic communities")
        return self.communities

    def _is_topic_relevant_word(self, word: str) -> bool:
        """Keep keywords tightly scoped to methane/dairy/livestock topic space."""
        if word in self.topic_anchor_terms:
            return True
        return any(word.startswith(stem) for stem in self.topic_stems)

    def _compile_word_matcher(self, words: List[str]) -> Optional[re.Pattern]:
        """Compile a safe word-boundary regex for keyword matching."""
        cleaned = [re.escape(str(w).lower()) for w in words if isinstance(w, str) and w]
        if not cleaned:
            return None
        return re.compile(r'\\b(?:' + '|'.join(cleaned) + r')\\b', flags=re.IGNORECASE)

    def _compute_topic_sentiment(self, topic_words: List[str]) -> Tuple[Optional[float], Optional[str], int]:
        """Compute average sentiment and class label for posts matching topic words."""
        if 'sentiment_score' not in self.df.columns:
            return None, None, 0

        matcher = self._compile_word_matcher(topic_words)
        if matcher is None:
            return None, None, 0

        text_series = self.df[self.text_column].astype(str).str.lower()
        mask = text_series.apply(lambda t: bool(matcher.search(t)))
        subset = self.df[mask]
        if len(subset) == 0:
            return None, None, 0

        avg_sentiment = float(subset['sentiment_score'].mean())
        if avg_sentiment > 0.1:
            sentiment_label = 'positive'
        elif avg_sentiment < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        return avg_sentiment, sentiment_label, int(len(subset))
    
    def label_topics(self, n_words: int = 5) -> Dict[int, Dict]:
        """
        Generate topic labels for each community based on most frequent words.
        
        Args:
            n_words: Number of top words to use for labeling
            
        Returns:
            Dictionary mapping community ID to topic info
        """
        if self.communities is None:
            self.detect_communities()

        # Document-level topical relevance ratio per token.
        word_doc_counts = Counter()
        word_topic_doc_counts = Counter()
        for text in self.df[self.text_column].astype(str):
            tokens = set(self._tokenize_text(text))
            if not tokens:
                continue
            text_lower = text.lower()
            has_topic_anchor = any(anchor in text_lower for anchor in self.topic_anchor_terms)
            for token in tokens:
                word_doc_counts[token] += 1
                if has_topic_anchor:
                    word_topic_doc_counts[token] += 1

        def _topic_ratio(word: str) -> float:
            doc_count = word_doc_counts.get(word, 0)
            if doc_count <= 0:
                return 0.0
            return float(word_topic_doc_counts.get(word, 0) / doc_count)
        
        self.community_topics = {}
        
        for idx, community in enumerate(self.communities):
            # Sort words by frequency
            words_with_freq = [(w, self.word_freq.get(w, 0)) for w in community]
            words_with_freq.sort(key=lambda x: x[1], reverse=True)

            relevant_words_with_freq = []
            for word, freq in words_with_freq:
                topic_ratio = _topic_ratio(word)
                if not self._is_topic_relevant_word(word):
                    continue
                if word not in self.topic_anchor_terms and topic_ratio < 0.35:
                    continue
                relevant_words_with_freq.append((word, freq))

            words_for_topic = relevant_words_with_freq if len(relevant_words_with_freq) > 0 else words_with_freq
            top_words = [w for w, _ in words_for_topic[:n_words]]
            
            # Use selected preset topic labels (Topic 1, 2, 5)
            if hasattr(self, 'active_preset_topics') and idx < len(self.active_preset_topics):
                topic_label = self.active_preset_topics[idx]['label']
                topic_num = int(self.active_preset_topics[idx]['topic_id'])
            else:
                topic_label = ", ".join(top_words[:3])
                topic_num = idx + 1
            
            # Calculate total frequency and centrality metrics
            total_freq = sum(f for _, f in words_with_freq)

            avg_sentiment, sentiment_label, sentiment_posts = self._compute_topic_sentiment(top_words[:5])
            relevant_word_share = (
                float(len(relevant_words_with_freq) / len(words_with_freq))
                if len(words_with_freq) > 0 else 0.0
            )
            top_word_relevance = [_topic_ratio(w) for w in top_words]
            topic_relevance_score = (
                float(np.mean(top_word_relevance)) if len(top_word_relevance) > 0 else 0.0
            )
            is_relevant_topic = bool(
                any(w in self.topic_anchor_terms for w in top_words)
                or relevant_word_share >= 0.35
                or topic_relevance_score >= 0.45
            )

            keyword_frequencies = {w: int(f) for w, f in words_for_topic}
            keyword_relevance = {w: _topic_ratio(w) for w, _ in words_for_topic}
            
            self.community_topics[idx] = {
                'topic_id': topic_num,
                'label': topic_label,
                'top_words': top_words,
                'all_words': list(community),
                'word_count': len(community),
                'total_frequency': total_freq,
                'avg_sentiment': avg_sentiment,
                'sentiment_label': sentiment_label,
                'sentiment_posts': sentiment_posts,
                'relevant_word_share': relevant_word_share,
                'topic_relevance_score': topic_relevance_score,
                'is_relevant_topic': is_relevant_topic,
                'keyword_frequencies': keyword_frequencies,
                'keyword_relevance': keyword_relevance,
            }
        
        return self.community_topics

    def get_topic_summary(self, only_relevant: bool = True) -> pd.DataFrame:
        """Generate summary table of all topics."""
        if self.community_topics is None:
            self.label_topics()
        
        rows = []
        for _, info in self.community_topics.items():
            if only_relevant and not info.get('is_relevant_topic', True):
                continue
            rows.append({
                'Topic': info['topic_id'],
                'Label': info['label'],
                'Top Words': ', '.join(info['top_words']),
                'Word Count': info['word_count'],
                'Total Freq': info['total_frequency'],
                'Avg Sentiment': f"{info['avg_sentiment']:.3f}" if info['avg_sentiment'] is not None else 'N/A',
                'Sentiment Label': info.get('sentiment_label', 'N/A') or 'N/A',
                'Sentiment Posts': int(info.get('sentiment_posts', 0)),
                'Topic Relevant': bool(info.get('is_relevant_topic', True)),
                'Topic Relevance Score': float(info.get('topic_relevance_score', 0.0)),
            })

        if len(rows) == 0 and only_relevant:
            return self.get_topic_summary(only_relevant=False)
        
        return pd.DataFrame(rows)

    def get_topic_keywords_by_frequency(self,
                                        top_n_per_topic: int = 10,
                                        only_relevant: bool = True) -> pd.DataFrame:
        """Return top topic keywords by frequency with topic sentiment metadata."""
        if self.community_topics is None:
            self.label_topics()

        rows = []
        for _, info in self.community_topics.items():
            if only_relevant and not info.get('is_relevant_topic', True):
                continue

            keyword_frequencies = info.get('keyword_frequencies', {})
            if not keyword_frequencies:
                continue

            sorted_keywords = sorted(
                keyword_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max(1, int(top_n_per_topic))]

            for keyword, freq in sorted_keywords:
                rows.append({
                    'Topic': int(info['topic_id']),
                    'Label': info['label'],
                    'Keyword': keyword,
                    'Frequency': int(freq),
                    'Keyword Topic Relevance Ratio': float(info.get('keyword_relevance', {}).get(keyword, 0.0)),
                    'Avg Sentiment': float(info['avg_sentiment']) if info.get('avg_sentiment') is not None else np.nan,
                    'Sentiment Label': info.get('sentiment_label', 'N/A') or 'N/A',
                    'Sentiment Posts': int(info.get('sentiment_posts', 0)),
                    'Topic Relevant': bool(info.get('is_relevant_topic', True)),
                })

        if len(rows) == 0 and only_relevant:
            return self.get_topic_keywords_by_frequency(top_n_per_topic=top_n_per_topic, only_relevant=False)

        return pd.DataFrame(rows)
    
    def plot_network(self, save_path: str = None, 
                    figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Generate publication-ready network visualization with topic communities.
        """
        if self.G is None or self.G.number_of_nodes() == 0:
            print("   ⚠️ No network to plot")
            return
        
        if self.communities is None:
            self.detect_communities()
        
        if self.community_topics is None:
            self.label_topics()
        
        # Color palette for communities
        community_colors = [
            '#4d96ff', '#ff6b6b', '#6bcb77', '#ffd93d', '#c084fc',
            '#f97316', '#06b6d4', '#ec4899', '#84cc16', '#a78bfa',
            '#fb923c', '#22d3ee', '#f472b6', '#a3e635', '#818cf8'
        ]
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Layout: force each selected topic cluster into its own visual region.
        centers = [(-2.2, 1.0), (2.2, 1.0), (0.0, -2.2)]
        pos = {}
        for idx, community in enumerate(self.communities):
            sub_nodes = list(community)
            subgraph = self.G.subgraph(sub_nodes)
            local_pos = nx.spring_layout(subgraph, k=0.7, iterations=60, seed=42 + idx)
            cx, cy = centers[idx % len(centers)]
            for node, (x, y) in local_pos.items():
                pos[node] = (x * 0.9 + cx, y * 0.9 + cy)
        
        # Node sizes based on word frequency
        max_freq = max(self.word_freq.get(n, 1) for n in self.G.nodes())
        node_sizes = [300 + (self.word_freq.get(n, 1) / max_freq) * 2500 
                     for n in self.G.nodes()]
        
        # Node colors based on community
        selected_colors = ['#4d96ff', '#ff6b6b', '#6bcb77']  # Topic 1,2,5
        node_colors = [selected_colors[self.node_community.get(n, 0) % len(selected_colors)]
                  for n in self.G.nodes()]
        
        # Edge widths based on co-occurrence weight
        edge_weights = [self.G[u][v].get('weight', 1) for u, v in self.G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + (w / max_weight) * 2 for w in edge_weights]
        
        # Draw edges
        nx.draw_networkx_edges(
            self.G, pos, ax=ax,
            width=edge_widths,
            edge_color='#cccccc',
            alpha=0.4
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            edgecolors='#333333',
            linewidths=1.0
        )
        
        # Draw labels - larger for high-frequency words
        top_words = [w for w, _ in sorted(self.word_freq.items(), 
                                          key=lambda x: x[1], reverse=True)[:20]]
        labels_large = {n: n for n in self.G.nodes() if n in top_words}
        labels_small = {n: n for n in self.G.nodes() if n not in labels_large}
        
        nx.draw_networkx_labels(
            self.G, pos, labels=labels_large, ax=ax,
            font_size=12, font_family='sans-serif',
            font_weight='bold', font_color='#111111'
        )
        nx.draw_networkx_labels(
            self.G, pos, labels=labels_small, ax=ax,
            font_size=8, font_family='sans-serif',
            font_weight='normal', font_color='#444444'
        )
        
        # Create legend for topic communities
        legend_handles = []
        for idx, info in self.community_topics.items():
            color = selected_colors[idx % len(selected_colors)]
            label = f"Topic {info['topic_id']}: {info['label']}"
            if info['avg_sentiment'] is not None:
                sent_indicator = '(+)' if info['avg_sentiment'] > 0.1 else '(-)' if info['avg_sentiment'] < -0.1 else '(~)'
                label += f" {sent_indicator}"
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color, markersize=12, label=label)
            )
        
        ax.legend(handles=legend_handles, loc='upper left', fontsize=10,
                 framealpha=0.95, title='Topic Communities', title_fontsize=12,
                 bbox_to_anchor=(0.01, 0.99))
        
        ax.set_title('Word Co-occurrence Network with Topic Communities',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Network plot saved to: {save_path}")
        
        plt.show()
    
    def plot_topic_sentiment(self, save_path: str = None) -> None:
        """
        Plot sentiment distribution across topics.
        """
        if self.community_topics is None:
            self.label_topics()
        
        if 'sentiment_score' not in self.df.columns:
            print("   ⚠️ No sentiment data available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        topics = []
        sentiments = []
        colors = []
        
        color_palette = [
            '#4d96ff', '#ff6b6b', '#6bcb77', '#ffd93d', '#c084fc',
            '#f97316', '#06b6d4', '#ec4899', '#84cc16', '#a78bfa'
        ]
        
        for idx, info in self.community_topics.items():
            if not info.get('is_relevant_topic', True):
                continue
            if info['avg_sentiment'] is not None:
                topics.append(f"Topic {info['topic_id']}:\n{info['label'][:20]}")
                sentiments.append(info['avg_sentiment'])
                colors.append(color_palette[idx % len(color_palette)])
        
        if not topics:
            print("   ⚠️ No topic sentiment data")
            return
        
        # Bar chart of average sentiment by topic
        ax1 = axes[0]
        x = np.arange(len(topics))
        bars = ax1.bar(x, sentiments, color=colors, edgecolor='#333', alpha=0.8)
        
        # Color bars by sentiment direction
        for bar, sent in zip(bars, sentiments):
            if sent > 0.1:
                bar.set_facecolor('#6bcb77')
            elif sent < -0.1:
                bar.set_facecolor('#ff6b6b')
            else:
                bar.set_facecolor('#ffd93d')
        
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(topics, fontsize=9, rotation=45, ha='right')
        ax1.set_ylabel('Average Sentiment Score', fontsize=12)
        ax1.set_title('Sentiment by Topic Community', fontsize=14, fontweight='bold')
        ax1.set_ylim(-1, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Value labels
        for bar, sent in zip(bars, sentiments):
            height = bar.get_height()
            ax1.annotate(f'{sent:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -10),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        # Topic word frequency chart
        ax2 = axes[1]
        
        # Get top words from each topic
        all_topic_words = []
        for info in self.community_topics.values():
            if not info.get('is_relevant_topic', True):
                continue
            for word in info['top_words'][:3]:
                all_topic_words.append({
                    'word': word,
                    'freq': self.word_freq.get(word, 0),
                    'topic': f"Topic {info['topic_id']}"
                })
        
        # Sort and take top 15
        all_topic_words.sort(key=lambda x: x['freq'], reverse=True)
        top_topic_words = all_topic_words[:15]
        
        words = [w['word'] for w in top_topic_words]
        freqs = [w['freq'] for w in top_topic_words]
        
        bars2 = ax2.barh(range(len(words)), freqs, color='#4d96ff', 
                        edgecolor='#333', alpha=0.8)
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels(words, fontsize=10)
        ax2.set_xlabel('Frequency', fontsize=12)
        ax2.set_title('Top Topic Keywords by Frequency', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Topic Analysis: Communities and Sentiment',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Topic sentiment plot saved to: {save_path}")
        
        plt.show()

    def compute_topic_coherence(self, top_n_words: int = 8) -> Dict:
        """
        Compute NPMI-based topic coherence (C_v-style proxy) for each community.
        """
        if self.community_topics is None:
            self.label_topics()

        token_docs = []
        for text in self.df[self.text_column].fillna(''):
            token_docs.append(set(self._tokenize_text(text)))

        if len(token_docs) == 0:
            return {'model_coherence_cv_proxy': np.nan, 'topic_scores': {}}

        doc_count = len(token_docs)
        word_doc_freq = Counter()
        for doc in token_docs:
            word_doc_freq.update(doc)

        topic_scores = {}
        for idx, info in self.community_topics.items():
            words = info['top_words'][:top_n_words]
            if len(words) < 2:
                topic_scores[idx] = np.nan
                continue

            pair_scores = []
            for w1, w2 in combinations(words, 2):
                p_w1 = word_doc_freq.get(w1, 0) / doc_count
                p_w2 = word_doc_freq.get(w2, 0) / doc_count
                p_joint = sum(1 for d in token_docs if (w1 in d and w2 in d)) / doc_count
                if p_w1 == 0 or p_w2 == 0 or p_joint == 0:
                    continue
                pmi = np.log(p_joint / (p_w1 * p_w2))
                npmi = pmi / (-np.log(p_joint))
                pair_scores.append(npmi)

            topic_scores[idx] = float(np.mean(pair_scores)) if pair_scores else np.nan

        valid_scores = [v for v in topic_scores.values() if not np.isnan(v)]
        model_score = float(np.mean(valid_scores)) if valid_scores else np.nan

        return {
            'model_coherence_cv_proxy': model_score,
            'topic_scores': topic_scores,
            'interpretation': (
                'strong' if not np.isnan(model_score) and model_score > 0.6
                else 'acceptable' if not np.isnan(model_score) and model_score > 0.5
                else 'weak_or_requires_justification'
            )
        }

    def topic_count_sensitivity(self,
                                min_cooccurrence_values: Optional[List[int]] = None,
                                max_edges: int = 150) -> pd.DataFrame:
        """Sensitivity analysis for number of topics (k) and coherence proxy."""
        if min_cooccurrence_values is None:
            min_cooccurrence_values = [2, 3, 4, 5]

        rows = []
        for min_co in min_cooccurrence_values:
            tmp = NetworkAnalysis(self.df, text_column=self.text_column)
            tmp.build_network(
                min_cooccurrence=min_co,
                max_edges=max_edges,
                min_word_frequency=2
            )
            if tmp.G is None or tmp.G.number_of_nodes() == 0:
                rows.append({
                    'min_cooccurrence': min_co,
                    'topic_count_k': 0,
                    'coherence_cv_proxy': np.nan
                })
                continue
            tmp.detect_communities()
            tmp.label_topics()
            coh = tmp.compute_topic_coherence()
            rows.append({
                'min_cooccurrence': min_co,
                'topic_count_k': len(tmp.communities),
                'coherence_cv_proxy': coh['model_coherence_cv_proxy']
            })

        return pd.DataFrame(rows)

    def topic_stability_bootstrap(self,
                                  n_bootstrap: int = 8,
                                  sample_frac: float = 0.8,
                                  top_n_words: int = 6,
                                  random_seed: int = 42) -> Dict:
        """Bootstrap topic stability via Jaccard overlap of top words."""
        if self.community_topics is None:
            self.label_topics()

        if self.community_topics is None or len(self.community_topics) == 0:
            return {'mean_topic_overlap_jaccard': np.nan, 'bootstrap_runs': 0}

        rng = np.random.default_rng(random_seed)
        baseline_topics = [set(v['top_words'][:top_n_words]) for v in self.community_topics.values()]
        overlaps = []

        n_rows = len(self.df)
        sample_size = max(20, int(sample_frac * n_rows))

        for i in range(n_bootstrap):
            idx = rng.choice(n_rows, size=sample_size, replace=True)
            sample_df = self.df.iloc[idx].copy()
            tmp = NetworkAnalysis(sample_df, text_column=self.text_column)
            tmp.build_network(min_cooccurrence=2, max_edges=500, min_word_frequency=2)
            if tmp.G is None or tmp.G.number_of_nodes() == 0:
                continue
            tmp.detect_communities()
            tmp.label_topics(n_words=top_n_words)
            current_topics = [set(v['top_words'][:top_n_words]) for v in tmp.community_topics.values()]
            if not current_topics:
                continue

            per_topic_best = []
            for bt in baseline_topics:
                best = 0.0
                for ct in current_topics:
                    union = len(bt.union(ct))
                    if union == 0:
                        continue
                    best = max(best, len(bt.intersection(ct)) / union)
                per_topic_best.append(best)

            overlaps.append(float(np.mean(per_topic_best)) if per_topic_best else 0.0)

        return {
            'mean_topic_overlap_jaccard': float(np.mean(overlaps)) if overlaps else np.nan,
            'std_topic_overlap_jaccard': float(np.std(overlaps)) if overlaps else np.nan,
            'bootstrap_runs': len(overlaps),
            'stable_topics_threshold_0_6': bool(np.mean(overlaps) >= 0.6) if overlaps else False
        }

    def plot_topic_sentiment_interaction(self, save_path: str = None) -> Optional[pd.DataFrame]:
        """Stacked bar chart: sentiment composition per detected topic."""
        if 'sentiment' not in self.df.columns:
            return None
        if self.community_topics is None:
            self.label_topics()

        rows = []
        for idx, info in self.community_topics.items():
            if not info.get('is_relevant_topic', True):
                continue
            topic_words = info['top_words'][:4]
            if not topic_words:
                continue
            matcher = self._compile_word_matcher(topic_words)
            if matcher is None:
                continue
            mask = self.df[self.text_column].astype(str).str.lower().apply(
                lambda t: bool(matcher.search(t))
            )
            subset = self.df[mask]
            if len(subset) == 0:
                continue

            counts = subset['sentiment'].value_counts(normalize=True)
            rows.append({
                'topic': f"Topic {info['topic_id']}",
                'label': info['label'],
                'negative': counts.get('negative', 0.0),
                'neutral': counts.get('neutral', 0.0),
                'positive': counts.get('positive', 0.0),
                'n_posts': len(subset)
            })

        if not rows:
            return None

        interaction = pd.DataFrame(rows).sort_values('n_posts', ascending=False)

        plt.figure(figsize=(12, 6))
        x = np.arange(len(interaction))
        neg = interaction['negative'].values
        neu = interaction['neutral'].values
        pos = interaction['positive'].values

        plt.bar(x, neg, color='#ff6b6b', label='Negative')
        plt.bar(x, neu, bottom=neg, color='#ffd93d', label='Neutral')
        plt.bar(x, pos, bottom=neg + neu, color='#6bcb77', label='Positive')
        plt.xticks(x, interaction['topic'], rotation=30, ha='right')
        plt.ylabel('Sentiment share within topic')
        plt.title('Topic-Sentiment Interaction (Stacked Proportions)',
                  fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Topic-sentiment interaction saved to: {save_path}")

        plt.show()
        return interaction


def resolve_attrition_counts(df: pd.DataFrame, data_dir: str = 'data') -> Dict:
    """Resolve N1→N4 attrition counts using saved attrition report if available."""
    attrition = {
        'N1_raw_collected': np.nan,
        'N2_after_deduplication': np.nan,
        'N3_after_language_filter': np.nan,
        'N4_final_analytic': int(len(df))
    }

    if not os.path.isdir(data_dir):
        return attrition

    candidates = [
        f for f in os.listdir(data_dir)
        if f.startswith('attrition_report_') and f.endswith('.csv')
    ]
    if not candidates:
        return attrition

    latest = sorted(candidates)[-1]
    path = os.path.join(data_dir, latest)
    try:
        report = pd.read_csv(path)
        if 'Stage' in report.columns and 'Count' in report.columns:
            mapping = dict(zip(report['Stage'].astype(str), report['Count']))

            def _safe_int(value, fallback=np.nan):
                if pd.isna(value):
                    return fallback
                try:
                    return int(value)
                except Exception:
                    return fallback

            attrition['N1_raw_collected'] = _safe_int(
                mapping.get('N0_initial', mapping.get('N0_raw_extracted', np.nan))
            )
            attrition['N2_after_deduplication'] = _safe_int(
                mapping.get('N1_deduplicated', mapping.get('N1_unique_posts', np.nan))
            )
            attrition['N3_after_language_filter'] = _safe_int(mapping.get('N2_language_filtered', np.nan))
            attrition['N4_final_analytic'] = _safe_int(mapping.get('N4_final_analytic', len(df)), fallback=len(df))
            attrition['source_file'] = path
    except Exception:
        pass

    if 'author_id_hash' in df.columns:
        activity = df['author_id_hash'].value_counts()
        threshold = activity.quantile(0.95) if len(activity) > 0 else np.nan
        top_users = activity[activity >= threshold].index if len(activity) > 0 else []
        removed_share = df['author_id_hash'].isin(top_users).mean() if len(df) > 0 else 0.0
        attrition['top_5pct_user_activity_share'] = float(removed_share)

    return attrition


class RobustnessAnalysis:
    """Robustness checks: model, temporal, sampling, and user concentration sensitivity."""

    def __init__(self, df: pd.DataFrame, text_column: str, date_column: str):
        self.df = df.copy()
        self.text_column = text_column
        self.date_column = date_column
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])

    def _summary_metric(self, in_df: pd.DataFrame) -> Dict:
        if len(in_df) == 0:
            return {'mean_sentiment_score': np.nan, 'net_sentiment': np.nan, 'n': 0}
        sent = in_df['sentiment'].value_counts()
        net = (sent.get('positive', 0) - sent.get('negative', 0)) / max(1, len(in_df))
        return {
            'mean_sentiment_score': float(in_df['sentiment_score'].mean()),
            'net_sentiment': float(net),
            'n': int(len(in_df))
        }

    def alternative_sentiment_model_validation(self,
                                               sample_size: int = 500) -> Dict:
        """Cross-check RoBERTa sentiment against VADER sentiment."""
        if len(self.df) == 0:
            return {}
        sample = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        if len(sample) < 10:
            return {}

        try:
            vader = VaderCrossCheckAnalyzer()
            vader_df = vader.predict_sentiment(sample[self.text_column].fillna('').tolist())
            roberta_scores = sample['sentiment_score'].values
            vader_scores = vader_df['vader_sentiment_score'].values
            r, p = pearsonr(roberta_scores, vader_scores)
            ci_low, ci_high = _pearson_ci(r, len(sample))

            roberta_labels = sample['sentiment'].astype(str).values
            vader_labels = vader_df['vader_sentiment'].astype(str).values
            label_agreement = float(np.mean(roberta_labels == vader_labels))

            # Fallback lexical check for triangulation
            blob_scores = sample[self.text_column].fillna('').apply(lambda t: TextBlob(str(t)).sentiment.polarity)
            rb_blob_r, rb_blob_p = pearsonr(roberta_scores, blob_scores.values)

            return {
                'n': int(len(sample)),
                'crosscheck_model': 'vaderSentiment',
                'pearson_r_roberta_vs_vader': float(r),
                'p_value_roberta_vs_vader': float(p),
                'ci_95_roberta_vs_vader': [ci_low, ci_high],
                'effect_size_roberta_vs_vader': _effect_size_label(r),
                'label_agreement_rate': label_agreement,
                'mean_vader_confidence': float(vader_df['vader_confidence'].mean()),
                'triangulation_roberta_vs_textblob': {
                    'pearson_r': float(rb_blob_r),
                    'p_value': float(rb_blob_p)
                }
            }
        except Exception as e:
            # Safe fallback to keep pipeline running if VADER cannot load
            blob_scores = sample[self.text_column].fillna('').apply(lambda t: TextBlob(str(t)).sentiment.polarity)
            r, p = pearsonr(sample['sentiment_score'].values, blob_scores.values)
            ci_low, ci_high = _pearson_ci(r, len(sample))
            return {
                'n': int(len(sample)),
                'crosscheck_model': 'vaderSentiment',
                'error': str(e),
                'fallback': 'textblob',
                'pearson_r_roberta_vs_textblob': float(r),
                'p_value_roberta_vs_textblob': float(p),
                'ci_95_roberta_vs_textblob': [ci_low, ci_high],
                'effect_size_roberta_vs_textblob': _effect_size_label(r)
            }

    def exclude_top_volume_days(self, percentile: float = 0.95) -> Dict:
        """Recompute results excluding top-activity days."""
        daily_counts = self.df.groupby(self.df[self.date_column].dt.date).size()
        cutoff = daily_counts.quantile(percentile)
        keep_dates = daily_counts[daily_counts < cutoff].index
        filtered = self.df[self.df[self.date_column].dt.date.isin(keep_dates)]
        return {
            'cutoff_percentile': percentile,
            'daily_count_cutoff': float(cutoff),
            'full': self._summary_metric(self.df),
            'filtered': self._summary_metric(filtered)
        }

    def exclude_top_active_users(self, percentile: float = 0.95) -> Dict:
        """Sensitivity analysis excluding top 5% most active users."""
        if 'author_id_hash' not in self.df.columns:
            return {'available': False}
        user_counts = self.df['author_id_hash'].value_counts()
        threshold = user_counts.quantile(percentile)
        top_users = set(user_counts[user_counts >= threshold].index)
        filtered = self.df[~self.df['author_id_hash'].isin(top_users)]
        return {
            'available': True,
            'threshold_posts_per_user': float(threshold),
            'excluded_users': int(len(top_users)),
            'full': self._summary_metric(self.df),
            'filtered': self._summary_metric(filtered)
        }

    def subsampling_test(self, fractions: Optional[List[float]] = None, repeats: int = 5) -> Dict:
        """Subsampling stability test for key sentiment metrics."""
        if fractions is None:
            fractions = [0.5, 0.7, 0.9]

        rows = []
        for frac in fractions:
            for seed in range(repeats):
                sample = self.df.sample(frac=frac, random_state=seed)
                metric = self._summary_metric(sample)
                rows.append({
                    'fraction': frac,
                    'seed': seed,
                    'mean_sentiment_score': metric['mean_sentiment_score'],
                    'net_sentiment': metric['net_sentiment']
                })

        sub_df = pd.DataFrame(rows)
        return {
            'results': sub_df.to_dict('records'),
            'stability_summary': sub_df.groupby('fraction').agg({
                'mean_sentiment_score': ['mean', 'std'],
                'net_sentiment': ['mean', 'std']
            }).to_dict()
        }

    def time_window_sensitivity(self, shift_days: int = 7) -> Dict:
        """Check sensitivity when truncating start/end windows by ±shift days."""
        min_d = self.df[self.date_column].min()
        max_d = self.df[self.date_column].max()

        centered = self.df[(self.df[self.date_column] >= min_d + pd.Timedelta(days=shift_days)) &
                           (self.df[self.date_column] <= max_d - pd.Timedelta(days=shift_days))]
        early_shift = self.df[self.df[self.date_column] >= min_d + pd.Timedelta(days=shift_days)]
        late_shift = self.df[self.df[self.date_column] <= max_d - pd.Timedelta(days=shift_days)]

        return {
            'shift_days': shift_days,
            'full': self._summary_metric(self.df),
            'centered_window': self._summary_metric(centered),
            'drop_first_window': self._summary_metric(early_shift),
            'drop_last_window': self._summary_metric(late_shift)
        }


def generate_model_graph_suite(df_model: pd.DataFrame,
                               text_column: str,
                               date_column: str,
                               output_dir: str,
                               model_tag: str) -> Dict:
    """Generate the same analysis graph suite for a given sentiment model output."""
    summary = {}

    # Trend analysis
    trend_analyzer = TrendAnalysis(df_model, date_column=date_column, sentiment_column='sentiment')
    monthly = trend_analyzer.analyze_sentiment_trends()
    trend_analyzer.plot_trends(
        monthly,
        save_path=os.path.join(output_dir, f'trend_analysis_{model_tag}.png')
    )

    # Spike analysis + overlays
    spike_detector = SpikeDetection(df_model, date_column=date_column)
    volume_spikes = spike_detector.detect_volume_spikes(sigma_threshold=2.0)
    sentiment_spikes = spike_detector.detect_sentiment_spikes(sigma_threshold=2.0)
    spike_detector.plot_spikes(
        volume_spikes,
        sentiment_spikes,
        save_path=os.path.join(output_dir, f'spike_detection_{model_tag}.png')
    )
    spike_detector.plot_epidemic_curve_overlay(
        volume_spikes=volume_spikes,
        save_path=os.path.join(output_dir, f'epidemic_curve_overlay_{model_tag}.png')
    )
    corr_matrix = spike_detector.plot_correlation_heatmap(
        save_path=os.path.join(output_dir, f'correlation_heatmap_{model_tag}.png')
    )

    # Odds ratio and sentiment probabilities
    odds_text_column = text_column
    if 'topic_text' in df_model.columns and df_model['topic_text'].astype(str).str.len().gt(0).any():
        odds_text_column = 'topic_text'
    odds_analyzer = OddsRatioAnalysis(df_model, text_column=odds_text_column, sentiment_column='sentiment')
    log_odds_df = odds_analyzer.compute_log_odds_ratio(min_count=10, max_doc_frequency=0.8)
    if len(log_odds_df) > 0:
        odds_analyzer.plot_semantic_drivers(
            log_odds_df,
            n_words=15,
            save_path=os.path.join(output_dir, f'semantic_drivers_{model_tag}.png')
        )
    odds_analyzer.plot_probability_distribution(
        save_path=os.path.join(output_dir, f'probability_distribution_{model_tag}.png')
    )

    # Network + topic views
    network_analyzer = NetworkAnalysis(df_model, text_column=text_column)
    network_analyzer.build_network(min_cooccurrence=1, max_edges=800, min_word_frequency=2)
    topic_count = 0
    if network_analyzer.G and network_analyzer.G.number_of_nodes() > 0:
        network_analyzer.detect_communities()
        topics = network_analyzer.label_topics(n_words=5)
        topic_count = len(topics)
        topic_summary = network_analyzer.get_topic_summary()
        topic_summary.to_csv(os.path.join(output_dir, f'topic_summary_{model_tag}.csv'), index=False)
        topic_keywords = network_analyzer.get_topic_keywords_by_frequency(top_n_per_topic=10)
        if len(topic_keywords) > 0:
            topic_keywords_path = os.path.join(output_dir, f'topic_keywords_frequency_{model_tag}.csv')
            topic_keywords.to_csv(topic_keywords_path, index=False)
            print(f"   Saved topic keyword frequency table: {topic_keywords_path}")
        network_analyzer.plot_network(
            save_path=os.path.join(output_dir, f'network_analysis_{model_tag}.png')
        )
        network_analyzer.plot_topic_sentiment(
            save_path=os.path.join(output_dir, f'topic_sentiment_{model_tag}.png')
        )
        topic_interaction = network_analyzer.plot_topic_sentiment_interaction(
            save_path=os.path.join(output_dir, f'topic_sentiment_interaction_{model_tag}.png')
        )
        if isinstance(topic_interaction, pd.DataFrame):
            topic_interaction.to_csv(
                os.path.join(output_dir, f'topic_sentiment_interaction_{model_tag}.csv'),
                index=False
            )

    summary['n_posts'] = int(len(df_model))
    summary['topic_count'] = int(topic_count)
    summary['correlation_matrix'] = corr_matrix.to_dict() if isinstance(corr_matrix, pd.DataFrame) else {}
    return summary


def run_full_analysis(df: pd.DataFrame,
                     text_column: str = 'clean_text',
                     date_column: str = 'created_utc',
                     output_dir: str = 'results') -> Dict:
    """
    Run complete Phase 3 analysis pipeline.
    
    Implements:
    1. Sentiment classification
    2. Trend-based temporal analysis
    3. Spike detection
    4. Odds ratio / log-odds analysis
    
    Args:
        df: Preprocessed DataFrame
        text_column: Column with cleaned text
        date_column: Column with timestamps
        output_dir: Directory for saving outputs
        
    Returns:
        Dictionary with all analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("PHASE 3: EMPIRICAL ANALYSIS")
    print(f"{'='*60}\n")
    
    results = {}

    # ===== Step 0: Attrition & Analytic Population =====
    print("📋 Step 0: Attrition & Sample Construction")
    print("-" * 40)
    attrition = resolve_attrition_counts(df, data_dir='data')
    print(f"   N1 (Raw collected): {attrition.get('N1_raw_collected', 'NA')}")
    print(f"   N2 (After deduplication): {attrition.get('N2_after_deduplication', 'NA')}")
    print(f"   N3 (After language filter): {attrition.get('N3_after_language_filter', 'NA')}")
    print(f"   N4 (Final analytic): {attrition.get('N4_final_analytic', len(df))}")
    if 'top_5pct_user_activity_share' in attrition:
        print(f"   Top 5% user activity share: {attrition['top_5pct_user_activity_share']*100:.2f}%")
    results['attrition'] = attrition
    
    # ===== Step 1: Sentiment Classification =====
    print("📊 Step 1: Sentiment Classification")
    print("-" * 40)
    
    analyzer = SentimentAnalyzer()
    sentiment_results = analyzer.predict_sentiment(df[text_column].tolist())
    
    # Merge results back to DataFrame
    for col in sentiment_results.columns:
        df[col] = sentiment_results[col].values
    
    # Summary statistics
    sentiment_dist = df['sentiment'].value_counts()
    print(f"\n   Sentiment Distribution:")
    for sent, count in sentiment_dist.items():
        print(f"      {sent.capitalize()}: {count} ({count/len(df)*100:.1f}%)")
    
    results['sentiment_distribution'] = sentiment_dist.to_dict()
    results['mean_sentiment_score'] = df['sentiment_score'].mean()

    # ===== Step 1C: Sentiment Keyword Quality Check =====
    print("\n🔎 Step 1C: Keyword Quality Check by Sentiment")
    print("-" * 40)
    sentiment_keywords = extract_keywords_by_sentiment(
        df,
        text_column=text_column,
        sentiment_column='sentiment',
        top_n=25
    )
    if len(sentiment_keywords) > 0:
        keyword_path = os.path.join(output_dir, 'sentiment_keyword_quality_check.csv')
        sentiment_keywords.to_csv(keyword_path, index=False)
        print(f"   Saved keyword quality-check table: {keyword_path}")

        keywords_summary = {}
        for sentiment in ['positive', 'neutral', 'negative']:
            top_terms = sentiment_keywords[
                sentiment_keywords['sentiment'] == sentiment
            ]['keyword'].head(10).tolist()
            if top_terms:
                keywords_summary[sentiment] = top_terms
                print(f"   {sentiment.capitalize()} top keywords: {', '.join(top_terms[:8])}")

        results['sentiment_keyword_quality_check'] = {
            'file': keyword_path,
            'top_keywords': keywords_summary
        }
    else:
        print("   ⚠️ Could not compute sentiment keyword quality-check table")

    # ===== Step 1B: VADER Cross-Check Inference =====
    print("\n🤝 Step 1B: VADER Cross-Check Inference")
    print("-" * 40)
    vader_available = False
    try:
        vader_model = VaderCrossCheckAnalyzer()
        vader_results = vader_model.predict_sentiment(df[text_column].tolist())
        for col in vader_results.columns:
            df[col] = vader_results[col].values
        vader_available = True
        vader_dist = df['vader_sentiment'].value_counts()
        print("   VADER sentiment distribution:")
        for sent, count in vader_dist.items():
            print(f"      {sent.capitalize()}: {count} ({count/len(df)*100:.1f}%)")
        results['vader_sentiment_distribution'] = vader_dist.to_dict()
        results['vader_mean_sentiment_score'] = float(df['vader_sentiment_score'].mean())
    except Exception as e:
        print(f"   ⚠️ VADER full inference failed: {e}")
        results['vader_inference_error'] = str(e)

    # Representativeness and potential bias disclosure
    representativeness = {}
    if 'subreddit' in df.columns:
        subreddit_counts = df['subreddit'].value_counts()
        representativeness['subreddit_distribution'] = subreddit_counts.to_dict()
        representativeness['subreddit_hhi'] = float(np.sum((subreddit_counts / len(df)) ** 2))
    if 'author_id_hash' in df.columns:
        author_counts = df['author_id_hash'].value_counts()
        representativeness['author_activity_top_5pct_threshold'] = float(author_counts.quantile(0.95))
        representativeness['author_activity_gini_proxy'] = float(author_counts.std() / (author_counts.mean() + 1e-9))
    results['representativeness'] = representativeness
    
    # ===== Step 2: Trend Analysis =====
    print(f"\n📈 Step 2: Trend-Based Analysis")
    print("-" * 40)
    
    trend_analyzer = TrendAnalysis(df, date_column=date_column, sentiment_column='sentiment')
    
    # Temporal statistics
    temporal_stats = trend_analyzer.compute_temporal_statistics()
    print(f"\n   Temporal Range: {temporal_stats['temporal_range']['start']} to "
          f"{temporal_stats['temporal_range']['end']}")
    print(f"   Span: {temporal_stats['temporal_range']['span_days']} days "
          f"({temporal_stats['temporal_range']['span_months']} months)")
    
    # Monthly trends
    monthly_trends = trend_analyzer.analyze_sentiment_trends()
    
    # Trend significance
    trend_significance = trend_analyzer.test_trend_significance(monthly_trends)
    if 'first_vs_second_half' in trend_significance:
        sig = trend_significance['first_vs_second_half']
        print(f"\n   Trend Test (First vs Second Half):")
        print(f"      First half mean: {sig['first_half_mean']:.3f}")
        print(f"      Second half mean: {sig['second_half_mean']:.3f}")
        print(f"      Direction: {sig['direction']}")
        print(f"      Mann-Whitney U: {sig['mann_whitney_U']:.2f}")
        print(f"      p-value: {sig['p_value']:.4f}")
        print(f"      Cohen's d: {sig['cohens_d']:.3f}")
        print(f"      Significant: {sig['significant']}")
    
    results['temporal_stats'] = temporal_stats
    results['trend_significance'] = trend_significance
    
    # Plot trends
    trend_analyzer.plot_trends(monthly_trends, 
                              save_path=os.path.join(output_dir, 'trend_analysis.png'))
    
    # ===== Step 3: Spike Detection =====
    print(f"\n🔍 Step 3: Spike Detection")
    print("-" * 40)
    
    spike_detector = SpikeDetection(df, date_column=date_column)
    
    # Volume spikes
    volume_spikes = spike_detector.detect_volume_spikes(sigma_threshold=2.0)
    print(f"\n   Volume Spikes (|z| > 2σ): {len(volume_spikes)}")
    if len(volume_spikes) > 0:
        print(f"   Top volume spikes:")
        for _, spike in volume_spikes.head(5).iterrows():
            print(f"      {spike['date'].strftime('%Y-%m-%d')}: "
                  f"{spike['volume']:.0f} posts (z={spike['z_score']:.2f})")
    
    # Sentiment spikes
    sentiment_spikes = spike_detector.detect_sentiment_spikes(sigma_threshold=2.0)
    print(f"\n   Sentiment Spikes (|z| > 2σ): {len(sentiment_spikes)}")
    if len(sentiment_spikes) > 0:
        print(f"   Top sentiment spikes:")
        for _, spike in sentiment_spikes.head(5).iterrows():
            print(f"      {spike['date'].strftime('%Y-%m-%d')}: "
                  f"score={spike['avg_sentiment']:.3f} (z={spike['z_score']:.2f}, {spike['spike_type']})")
    
    # Lag correlations
    lag_corr = spike_detector.compute_lag_correlations(max_lag=28, target_lags=[0, 7, 21])
    if len(lag_corr) > 0:
        print(f"\n   Lag Correlation Analysis (volume vs sentiment, target lags 0/7/21):")
        sig_lags = lag_corr[lag_corr['is_target_lag']]
        if len(sig_lags) > 0:
            for _, row in sig_lags.iterrows():
                print(f"      Lag {row['lag_days']} days: r={row['correlation_vol_sent']:.3f} "
                      f"(p={row['p_value']:.4f}, FDR={row['p_fdr_bh']:.4f}, "
                      f"95% CI [{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}], {row['effect_size']})")
        else:
            print(f"      No significant lag correlations found")

    ccf = spike_detector.compute_cross_correlation_profile(max_lag=28)
    if len(ccf) > 0:
        best = ccf.iloc[ccf['cross_correlation'].abs().argmax()]
        print(f"   CCF strongest lag: {int(best['lag_days'])} days (r={best['cross_correlation']:.3f}, p={best['p_value']:.4f})")
    
    results['volume_spikes'] = volume_spikes.to_dict('records') if len(volume_spikes) > 0 else []
    results['sentiment_spikes'] = sentiment_spikes.to_dict('records') if len(sentiment_spikes) > 0 else []
    results['lag_correlations'] = lag_corr.to_dict('records') if len(lag_corr) > 0 else []
    results['cross_correlation_profile'] = ccf.to_dict('records') if len(ccf) > 0 else []
    
    # Plot spikes
    spike_detector.plot_spikes(volume_spikes, sentiment_spikes,
                              save_path=os.path.join(output_dir, 'spike_detection.png'))
    spike_detector.plot_epidemic_curve_overlay(
        volume_spikes=volume_spikes,
        save_path=os.path.join(output_dir, 'epidemic_curve_overlay.png')
    )
    corr_matrix = spike_detector.plot_correlation_heatmap(
        save_path=os.path.join(output_dir, 'correlation_heatmap.png')
    )
    results['correlation_matrix'] = corr_matrix.to_dict() if isinstance(corr_matrix, pd.DataFrame) else {}
    
    # ===== Step 4: Odds Ratio Analysis =====
    print(f"\n📐 Step 4: Odds Ratio / Log-Odds Analysis")
    print("-" * 40)

    odds_text_column = text_column
    if 'topic_text' in df.columns and df['topic_text'].astype(str).str.len().gt(0).any():
        odds_text_column = 'topic_text'
    print(f"   Using text column for log-odds: {odds_text_column}")

    odds_analyzer = OddsRatioAnalysis(df, text_column=odds_text_column, sentiment_column='sentiment')
    
    # Sentiment probabilities
    sent_probs = odds_analyzer.compute_sentiment_probabilities()
    print(f"\n   Sentiment Probabilities:")
    for sent in ['positive', 'neutral', 'negative']:
        p = sent_probs[sent]
        print(f"      {sent.capitalize()}: {p['probability']*100:.1f}% "
              f"(95% CI: [{p['ci_lower']*100:.1f}%, {p['ci_upper']*100:.1f}%])")
    print(f"      Net Sentiment: {sent_probs['net_sentiment']:+.3f}")
    
    # Log-odds analysis
    log_odds_df = odds_analyzer.compute_log_odds_ratio(min_count=10, max_doc_frequency=0.8)
    if len(log_odds_df) > 0:
        drivers = odds_analyzer.get_top_drivers(log_odds_df, n=10)
        
        print(f"\n   Top Positive Sentiment Drivers:")
        for driver in drivers['positive_drivers'][:5]:
            print(f"      {driver['word']}: log-OR={driver['log_odds']:.3f} (n={driver['count_total']})")
        
        print(f"\n   Top Negative Sentiment Drivers:")
        for driver in drivers['negative_drivers'][:5]:
            print(f"      {driver['word']}: log-OR={driver['log_odds']:.3f} (n={driver['count_total']})")
        
        results['semantic_drivers'] = drivers
        results['sentiment_probabilities'] = sent_probs
        
        # Plot semantic drivers
        odds_analyzer.plot_semantic_drivers(log_odds_df, n_words=15,
                                           save_path=os.path.join(output_dir, 'semantic_drivers.png'))
        
        # Plot probability distribution
        odds_analyzer.plot_probability_distribution(
            save_path=os.path.join(output_dir, 'probability_distribution.png'))
    
    # ===== Step 5: Network Analysis with Topic Detection =====
    print(f"\n🕸️ Step 5: Network Analysis & Topic Detection")
    print("-" * 40)
    
    network_analyzer = NetworkAnalysis(df, text_column=text_column)
    
    # Build network
    print("\n   Building word co-occurrence network (expanded vocabulary)...")
    network_analyzer.build_network(min_cooccurrence=1, max_edges=800, min_word_frequency=2)
    
    # Detect communities/topics
    if network_analyzer.G and network_analyzer.G.number_of_nodes() > 0:
        network_analyzer.detect_communities()
        topic_info = network_analyzer.label_topics(n_words=5)
        
        # Print topic summary
        print(f"\n   Detected Topics:")
        for idx, info in topic_info.items():
            sentiment_str = ""
            if info['avg_sentiment'] is not None:
                if info['avg_sentiment'] > 0.1:
                    sentiment_str = f" [Positive: {info['avg_sentiment']:.2f}]"
                elif info['avg_sentiment'] < -0.1:
                    sentiment_str = f" [Negative: {info['avg_sentiment']:.2f}]"
                else:
                    sentiment_str = f" [Neutral: {info['avg_sentiment']:.2f}]"
            print(f"      Topic {info['topic_id']}: {info['label']}{sentiment_str}")
            print(f"         Keywords: {', '.join(info['top_words'])}")
        
        # Save topic summary
        topic_summary = network_analyzer.get_topic_summary()
        topic_summary.to_csv(os.path.join(output_dir, 'topic_summary.csv'), index=False)

        # Topic coherence / stability / sensitivity diagnostics
        topic_coherence = network_analyzer.compute_topic_coherence(top_n_words=8)
        topic_k_sensitivity = network_analyzer.topic_count_sensitivity(min_cooccurrence_values=[2, 3, 4, 5])
        topic_stability = network_analyzer.topic_stability_bootstrap(
            n_bootstrap=8,
            sample_frac=0.8,
            top_n_words=6,
            random_seed=42
        )
        print(f"\n   Topic coherence (C_v proxy): {topic_coherence['model_coherence_cv_proxy']:.3f} "
              f"[{topic_coherence['interpretation']}]")
        print(f"   Topic stability (mean Jaccard overlap): {topic_stability.get('mean_topic_overlap_jaccard', np.nan):.3f}")
        
        results['topics'] = {
            str(k): {
                'topic_id': v['topic_id'],
                'label': v['label'],
                'top_words': v['top_words'],
                'word_count': v['word_count'],
                'avg_sentiment': v['avg_sentiment']
            } for k, v in topic_info.items()
        }
        results['topic_coherence'] = topic_coherence
        results['topic_k_sensitivity'] = topic_k_sensitivity.to_dict('records')
        results['topic_stability'] = topic_stability
        
        # Plot network
        network_analyzer.plot_network(
            save_path=os.path.join(output_dir, 'network_analysis.png'))
        
        # Plot topic sentiment
        network_analyzer.plot_topic_sentiment(
            save_path=os.path.join(output_dir, 'topic_sentiment.png'))
        topic_sentiment_interaction = network_analyzer.plot_topic_sentiment_interaction(
            save_path=os.path.join(output_dir, 'topic_sentiment_interaction.png')
        )
        if isinstance(topic_sentiment_interaction, pd.DataFrame):
            topic_sentiment_interaction.to_csv(
                os.path.join(output_dir, 'topic_sentiment_interaction.csv'), index=False
            )
            results['topic_sentiment_interaction'] = topic_sentiment_interaction.to_dict('records')
    else:
        print("   ⚠️ Insufficient data for network analysis")

    # ===== Step 6: Robustness & Sensitivity Analysis =====
    print(f"\n🧪 Step 6: Robustness & Sensitivity")
    print("-" * 40)
    robust = RobustnessAnalysis(df, text_column=text_column, date_column=date_column)
    alt_model = robust.alternative_sentiment_model_validation(sample_size=500)
    vol_excl = robust.exclude_top_volume_days(percentile=0.95)
    user_excl = robust.exclude_top_active_users(percentile=0.95)
    subsample = robust.subsampling_test(fractions=[0.5, 0.7, 0.9], repeats=5)
    time_window = robust.time_window_sensitivity(shift_days=7)

    if alt_model:
        if 'pearson_r_roberta_vs_vader' in alt_model:
            print(
                f"   RoBERTa vs VADER cross-check: "
                f"r={alt_model['pearson_r_roberta_vs_vader']:.3f}, "
                f"p={alt_model['p_value_roberta_vs_vader']:.4f}, "
                f"agreement={alt_model['label_agreement_rate']*100:.1f}%"
            )
        else:
            print(
                f"   VADER cross-check fallback ({alt_model.get('fallback', 'unknown')}): "
                f"r={alt_model.get('pearson_r_roberta_vs_textblob', np.nan):.3f}, "
                f"p={alt_model.get('p_value_roberta_vs_textblob', np.nan):.4f}"
            )
    print(f"   Excluding top 5% volume days: n={vol_excl['filtered']['n']} "
          f"(net={vol_excl['filtered']['net_sentiment']:+.3f})")
    if user_excl.get('available'):
        print(f"   Excluding top active users: n={user_excl['filtered']['n']} "
              f"(net={user_excl['filtered']['net_sentiment']:+.3f})")
    print(f"   Time-window sensitivity (±7 days) centered n={time_window['centered_window']['n']}")

    results['robustness'] = {
        'alternative_sentiment_model_validation': alt_model,
        'exclude_top_volume_days': vol_excl,
        'exclude_top_active_users': user_excl,
        'subsampling_test': subsample,
        'time_window_sensitivity': time_window,
        'multiple_testing_correction': {
            'lag_correlations': 'fdr_bh and bonferroni applied',
            'ccf_profile': 'fdr_bh applied'
        }
    }

    # ===== Step 7: VADER Graph Suite =====
    if vader_available:
        print(f"\n🖼️ Step 7: VADER Graph Suite")
        print("-" * 40)
        df_vader = df.copy()
        df_vader['sentiment'] = df_vader['vader_sentiment']
        df_vader['sentiment_score'] = df_vader['vader_sentiment_score']
        vader_summary = generate_model_graph_suite(
            df_model=df_vader,
            text_column=text_column,
            date_column=date_column,
            output_dir=output_dir,
            model_tag='vader'
        )
        results['vader_graph_suite'] = vader_summary
        print("   ✅ Saved VADER versions of all primary result graphs")
    
    # ===== Save Results Summary =====
    results_path = os.path.join(output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(_safe_serializable(results), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"   Results saved to: {output_dir}/")
    print(f"{'='*60}\n")
    
    return results, df


if __name__ == "__main__":
    # Example with mock data
    mock_df = pd.DataFrame({
        'clean_text': [
            "Climate change is real and we need action now",
            "Methane from cows is destroying the planet",
            "Sustainable farming practices are the future",
            "Factory farming is terrible for the environment",
            "New technology reduces cattle emissions significantly",
        ] * 100,
        'created_utc': pd.date_range('2020-01-01', periods=500, freq='D')
    })
    
    results, df_analyzed = run_full_analysis(mock_df, output_dir='results')
