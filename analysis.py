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
from scipy import stats
from scipy.special import softmax
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, spearmanr
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch

warnings.filterwarnings('ignore')


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
        
        stats = {
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
            'posts_per_year': df['year'].value_counts().sort_index().to_dict()
        }
        
        return stats
    
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
                pooled_std = np.sqrt((np.var(first_half) + np.var(second_half)) / 2)
                cohens_d = (np.mean(second_half) - np.mean(first_half)) / pooled_std if pooled_std > 0 else 0
                
                results['first_vs_second_half'] = {
                    'first_half_mean': np.mean(first_half),
                    'second_half_mean': np.mean(second_half),
                    'mann_whitney_U': stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'direction': 'increasing' if np.mean(second_half) > np.mean(first_half) else 'decreasing',
                    'significant': p_value < 0.05
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
            lambda x: 'p<0.05' if x > 1.96 else ('p<0.01' if x > 2.58 else 'p<0.001' if x > 3.29 else 'ns')
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
                                 max_lag: int = 14) -> pd.DataFrame:
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
                
                results.append({
                    'lag_days': lag,
                    'correlation_vol_sent': r_vol_sent,
                    'p_value': p_vol_sent,
                    'significant': p_vol_sent < 0.05,
                    'n': len(df_lag)
                })
        
        return pd.DataFrame(results)
    
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
    
    def compute_word_frequencies(self) -> Tuple[Counter, Counter, Counter]:
        """Compute word frequencies by sentiment class."""
        from collections import Counter
        import re
        
        positive_words = Counter()
        negative_words = Counter()
        all_words = Counter()
        
        for _, row in self.df.iterrows():
            text = str(row[self.text_column]).lower()
            words = re.findall(r'\b[a-z]{3,}\b', text)
            sentiment = row[self.sentiment_column]
            
            all_words.update(words)
            
            if sentiment == 'positive':
                positive_words.update(words)
            elif sentiment == 'negative':
                negative_words.update(words)
        
        return positive_words, negative_words, all_words
    
    def compute_log_odds_ratio(self, min_count: int = 10,
                               prior_count: float = 0.5) -> pd.DataFrame:
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
        
        # Total counts
        total_positive = sum(positive_words.values())
        total_negative = sum(negative_words.values())
        
        results = []
        
        for word, total_count in all_words.items():
            if total_count < min_count:
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
                'sentiment_driver': 'positive' if log_odds > 0 else 'negative'
            })
        
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
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Extract clean tokens from text."""
        text = str(text).lower()
        # Extract words (alphanumeric, 3+ chars)
        tokens = re.findall(r'\b[a-z]{3,}\b', text)
        # Filter stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        return tokens
    
    def build_network(self, min_cooccurrence: int = 3, 
                     max_edges: int = 150,
                     window_size: int = None) -> nx.Graph:
        """
        Build word co-occurrence network.
        
        Args:
            min_cooccurrence: Minimum co-occurrence count for edge
            max_edges: Maximum number of edges to include
            window_size: If set, only count co-occurrences within window
                        If None, use document-level co-occurrence
        
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
        
        edges_added = 0
        for (w1, w2), weight in co_occurrence.most_common():
            if weight < min_cooccurrence:
                break
            if edges_added >= max_edges:
                break
            self.G.add_edge(w1, w2, weight=weight)
            edges_added += 1
        
        print(f"   Network: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G
    
    def detect_communities(self) -> List[set]:
        """
        Detect topic communities using greedy modularity optimization.
        
        Returns:
            List of communities (sets of words)
        """
        if self.G is None or self.G.number_of_nodes() == 0:
            print("   ⚠️ No network built yet")
            return []
        
        from networkx.algorithms.community import greedy_modularity_communities
        
        self.communities = list(greedy_modularity_communities(self.G, weight='weight'))
        
        # Map nodes to community IDs
        self.node_community = {}
        for idx, community in enumerate(self.communities):
            for node in community:
                self.node_community[node] = idx
        
        print(f"   Detected {len(self.communities)} topic communities")
        return self.communities
    
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
        
        self.community_topics = {}
        
        for idx, community in enumerate(self.communities):
            # Sort words by frequency
            words_with_freq = [(w, self.word_freq.get(w, 0)) for w in community]
            words_with_freq.sort(key=lambda x: x[1], reverse=True)
            
            top_words = [w for w, _ in words_with_freq[:n_words]]
            
            # Generate topic label from top 3 words
            topic_label = ", ".join(top_words[:3])
            
            # Calculate total frequency and centrality metrics
            total_freq = sum(f for _, f in words_with_freq)
            
            # Get average sentiment if available
            avg_sentiment = None
            if 'sentiment_score' in self.df.columns:
                # Find posts containing any of the top words
                mask = self.df[self.text_column].str.lower().str.contains(
                    '|'.join(top_words[:3]), regex=True, na=False
                )
                if mask.sum() > 0:
                    avg_sentiment = self.df.loc[mask, 'sentiment_score'].mean()
            
            self.community_topics[idx] = {
                'topic_id': idx + 1,
                'label': topic_label,
                'top_words': top_words,
                'all_words': list(community),
                'word_count': len(community),
                'total_frequency': total_freq,
                'avg_sentiment': avg_sentiment
            }
        
        return self.community_topics
    
    def get_topic_summary(self) -> pd.DataFrame:
        """Generate summary table of all topics."""
        if self.community_topics is None:
            self.label_topics()
        
        rows = []
        for topic_id, info in self.community_topics.items():
            rows.append({
                'Topic': info['topic_id'],
                'Label': info['label'],
                'Top Words': ', '.join(info['top_words']),
                'Word Count': info['word_count'],
                'Total Freq': info['total_frequency'],
                'Avg Sentiment': f"{info['avg_sentiment']:.3f}" if info['avg_sentiment'] else 'N/A'
            })
        
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
        
        # Layout
        pos = nx.spring_layout(self.G, k=0.6, iterations=50, seed=42)
        
        # Node sizes based on word frequency
        max_freq = max(self.word_freq.get(n, 1) for n in self.G.nodes())
        node_sizes = [300 + (self.word_freq.get(n, 1) / max_freq) * 2500 
                     for n in self.G.nodes()]
        
        # Node colors based on community
        node_colors = [community_colors[self.node_community.get(n, 0) % len(community_colors)] 
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
            color = community_colors[idx % len(community_colors)]
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
    lag_corr = spike_detector.compute_lag_correlations(max_lag=14)
    if len(lag_corr) > 0:
        print(f"\n   Lag Correlation Analysis (volume vs sentiment):")
        sig_lags = lag_corr[lag_corr['significant']]
        if len(sig_lags) > 0:
            for _, row in sig_lags.iterrows():
                print(f"      Lag {row['lag_days']} days: r={row['correlation_vol_sent']:.3f} "
                      f"(p={row['p_value']:.4f})")
        else:
            print(f"      No significant lag correlations found")
    
    results['volume_spikes'] = volume_spikes.to_dict('records') if len(volume_spikes) > 0 else []
    results['sentiment_spikes'] = sentiment_spikes.to_dict('records') if len(sentiment_spikes) > 0 else []
    results['lag_correlations'] = lag_corr.to_dict('records') if len(lag_corr) > 0 else []
    
    # Plot spikes
    spike_detector.plot_spikes(volume_spikes, sentiment_spikes,
                              save_path=os.path.join(output_dir, 'spike_detection.png'))
    
    # ===== Step 4: Odds Ratio Analysis =====
    print(f"\n📐 Step 4: Odds Ratio / Log-Odds Analysis")
    print("-" * 40)
    
    odds_analyzer = OddsRatioAnalysis(df, text_column=text_column, sentiment_column='sentiment')
    
    # Sentiment probabilities
    sent_probs = odds_analyzer.compute_sentiment_probabilities()
    print(f"\n   Sentiment Probabilities:")
    for sent in ['positive', 'neutral', 'negative']:
        p = sent_probs[sent]
        print(f"      {sent.capitalize()}: {p['probability']*100:.1f}% "
              f"(95% CI: [{p['ci_lower']*100:.1f}%, {p['ci_upper']*100:.1f}%])")
    print(f"      Net Sentiment: {sent_probs['net_sentiment']:+.3f}")
    
    # Log-odds analysis
    log_odds_df = odds_analyzer.compute_log_odds_ratio(min_count=10)
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
    print("\n   Building word co-occurrence network...")
    network_analyzer.build_network(min_cooccurrence=3, max_edges=150)
    
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
        
        results['topics'] = {
            str(k): {
                'topic_id': v['topic_id'],
                'label': v['label'],
                'top_words': v['top_words'],
                'word_count': v['word_count'],
                'avg_sentiment': v['avg_sentiment']
            } for k, v in topic_info.items()
        }
        
        # Plot network
        network_analyzer.plot_network(
            save_path=os.path.join(output_dir, 'network_analysis.png'))
        
        # Plot topic sentiment
        network_analyzer.plot_topic_sentiment(
            save_path=os.path.join(output_dir, 'topic_sentiment.png'))
    else:
        print("   ⚠️ Insufficient data for network analysis")
    
    # ===== Save Results Summary =====
    import json
    
    # Convert non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj
    
    results_path = os.path.join(output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    
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
