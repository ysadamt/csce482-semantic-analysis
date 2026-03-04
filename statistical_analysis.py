"""
Phase 4: Publication-Ready Statistical Analysis
Covers every checklist item required for peer-reviewed reporting:

1. Data & Relevance  -- DataQualityReport
2. Hypothesis & Statistical Testing  -- HypothesisTestingSuite
3. Bias Assessment  -- BiasAssessment
4. Sensitivity Analysis (top-5 % users)  -- SensitivityAnalysis
5. Drivers of Sentiment (logistic regression)  -- SentimentDriverAnalysis

All functions accept and return plain dicts / DataFrames so they
integrate seamlessly with the existing pipeline.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    mannwhitneyu,
    shapiro,
    ttest_ind,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return 0.0 if pooled == 0 else float((np.mean(a) - np.mean(b)) / pooled)


def _ci_mean_diff(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Welch-approximation 95 % CI for the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = float(np.mean(a) - np.mean(b))
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    if se == 0:
        return diff, diff
    df_welch = (np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)) ** 2 / (
        (np.var(a, ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (np.var(b, ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df=max(1, df_welch))
    return float(diff - t_crit * se), float(diff + t_crit * se)


def _safe_serializable(obj: Any) -> Any:
    """Recursively convert numpy / pandas objects to JSON-friendly types."""
    if obj is pd.NA or obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_safe_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _safe_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serializable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    return str(obj)


# ============================================================================
# 1. DATA & RELEVANCE
# ============================================================================

class DataQualityReport:
    """
    Confirm the dataset is clean, relevant, and comparison-groups are defined.

    Checks performed:
    - Exact-duplicate detection (on clean_text)
    - Missing-value summary per column
    - Relevance verification (keyword hit-rate in text)
    - Comparison-group construction (Before / After a policy date)
    """

    DOMAIN_KEYWORDS = [
        "methane", "dairy", "cow", "cows", "cattle", "livestock",
        "emission", "emissions", "climate", "greenhouse",
        "farming", "farm", "enteric", "manure",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "clean_text",
        date_column: str = "created_utc",
        policy_date: str = "2024-01-01",
    ):
        self.df = df.copy()
        self.text_col = text_column
        self.date_col = date_column
        self.policy_date = pd.to_datetime(policy_date)
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

    # ------------------------------------------------------------------
    def check_duplicates(self) -> Dict:
        n_total = len(self.df)
        n_dup = self.df.duplicated(subset=[self.text_col], keep="first").sum()
        return {
            "total_rows": n_total,
            "exact_duplicates": int(n_dup),
            "duplicate_pct": round(n_dup / n_total * 100, 2) if n_total else 0,
            "clean": n_dup == 0,
        }

    def check_missing_values(self) -> Dict:
        missing = self.df.isnull().sum()
        cols_with_missing = missing[missing > 0].to_dict()
        return {
            "columns_with_missing": {k: int(v) for k, v in cols_with_missing.items()},
            "total_missing_cells": int(missing.sum()),
            "clean": int(missing.sum()) == 0,
        }

    def check_relevance(self) -> Dict:
        """Fraction of posts containing ≥1 domain keyword."""
        pattern = "|".join(self.DOMAIN_KEYWORDS)
        hits = self.df[self.text_col].str.lower().str.contains(pattern, na=False)
        hit_count = int(hits.sum())
        return {
            "keyword_hit_count": hit_count,
            "keyword_hit_pct": round(hit_count / len(self.df) * 100, 2) if len(self.df) else 0,
            "keywords_checked": self.DOMAIN_KEYWORDS,
            "relevant": hit_count / max(1, len(self.df)) > 0.50,
        }

    def define_comparison_groups(self) -> Dict:
        """
        Split dataset into 'before' and 'after' groups around the policy date.
        Returns counts and date ranges for each group.
        """
        before = self.df[self.df[self.date_col] < self.policy_date]
        after = self.df[self.df[self.date_col] >= self.policy_date]
        return {
            "policy_date": str(self.policy_date.date()),
            "before": {
                "n": len(before),
                "date_range": [
                    str(before[self.date_col].min().date()) if len(before) else None,
                    str(before[self.date_col].max().date()) if len(before) else None,
                ],
            },
            "after": {
                "n": len(after),
                "date_range": [
                    str(after[self.date_col].min().date()) if len(after) else None,
                    str(after[self.date_col].max().date()) if len(after) else None,
                ],
            },
            "balanced": 0.33 < len(before) / max(1, len(before) + len(after)) < 0.67,
        }

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """Run all data-quality checks and return composite report."""
        report = {
            "duplicates": self.check_duplicates(),
            "missing_values": self.check_missing_values(),
            "relevance": self.check_relevance(),
            "comparison_groups": self.define_comparison_groups(),
        }
        report["all_clean"] = (
            report["duplicates"]["clean"]
            and report["missing_values"]["clean"]
            and report["relevance"]["relevant"]
        )
        return report


# ============================================================================
# 2. HYPOTHESIS & STATISTICAL TESTING
# ============================================================================

class HypothesisTestingSuite:
    """
    Formal hypothesis testing for Before vs After comparison.

    Workflow:
    1. State H₀ and H₁.
    2. Test normality (Shapiro-Wilk) in both groups.
    3. Select independent t-test (normal) or Mann-Whitney U (non-normal).
    4. Report test statistic, p-value, Cohen's d, 95 % CI.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        score_column: str = "sentiment_score",
        date_column: str = "created_utc",
        policy_date: str = "2024-01-01",
    ):
        self.df = df.copy()
        self.score_col = score_column
        self.date_col = date_column
        self.policy_date = pd.to_datetime(policy_date)
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        self.before = self.df.loc[
            self.df[self.date_col] < self.policy_date, self.score_col
        ].dropna().values
        self.after = self.df.loc[
            self.df[self.date_col] >= self.policy_date, self.score_col
        ].dropna().values

    # ------------------------------------------------------------------
    def state_hypotheses(self) -> Dict:
        return {
            "H0": (
                "There is no significant difference in mean sentiment score "
                "between the Before-policy and After-policy periods "
                "(μ_before = μ_after)."
            ),
            "H1": (
                "There is a significant difference in mean sentiment score "
                "between the Before-policy and After-policy periods "
                "(μ_before ≠ μ_after)."
            ),
            "alpha": 0.05,
            "two_tailed": True,
        }

    def test_normality(self, max_sample: int = 5000) -> Dict:
        """
        Shapiro-Wilk normality test on both groups.
        For large N the test is applied to a random subsample (Shapiro-Wilk
        is limited to n ≤ 5000).
        """
        result = {}
        for label, arr in [("before", self.before), ("after", self.after)]:
            n = len(arr)
            if n < 3:
                result[label] = {"n": n, "normal": False, "note": "too few observations"}
                continue
            sample = arr if n <= max_sample else np.random.default_rng(42).choice(arr, max_sample, replace=False)
            stat, p = shapiro(sample)
            result[label] = {
                "n": n,
                "shapiro_W": round(float(stat), 6),
                "p_value": float(p),
                "normal_at_0_05": bool(p > 0.05),
                "sample_used": min(n, max_sample),
            }
        result["both_normal"] = all(
            result.get(g, {}).get("normal_at_0_05", False) for g in ("before", "after")
        )
        return result

    def select_and_run_test(self, normality: Dict) -> Dict:
        """
        Automatically choose the right test based on normality results.

        - If both groups are normal → Independent t-test (Welch's).
        - Otherwise → Mann-Whitney U test.

        Reports: test name, statistic (t or U), p-value, Cohen's d, 95 % CI
        for the mean difference.
        """
        if len(self.before) < 2 or len(self.after) < 2:
            return {"error": "Insufficient data in one or both groups."}

        use_ttest = normality.get("both_normal", False)

        if use_ttest:
            t_stat, p_value = ttest_ind(self.before, self.after, equal_var=False)
            test_name = "Welch's Independent t-test"
            stat_label = "t"
            stat_value = float(t_stat)
        else:
            u_stat, p_value = mannwhitneyu(self.before, self.after, alternative="two-sided")
            test_name = "Mann-Whitney U test"
            stat_label = "U"
            stat_value = float(u_stat)

        d = _cohens_d(self.before, self.after)
        ci_lo, ci_hi = _ci_mean_diff(self.before, self.after)

        # Effect-size interpretation (Cohen 1988)
        abs_d = abs(d)
        if abs_d < 0.2:
            d_interp = "negligible"
        elif abs_d < 0.5:
            d_interp = "small"
        elif abs_d < 0.8:
            d_interp = "medium"
        else:
            d_interp = "large"

        return {
            "test_name": test_name,
            "test_selected_because": (
                "Both groups passed Shapiro-Wilk normality test (p > .05)"
                if use_ttest
                else "At least one group failed Shapiro-Wilk normality test (p ≤ .05)"
            ),
            "statistic_label": stat_label,
            "statistic_value": round(stat_value, 4),
            "p_value": float(p_value),
            "significant_at_0_05": bool(p_value < 0.05),
            "cohens_d": round(d, 4),
            "cohens_d_interpretation": d_interp,
            "mean_before": round(float(np.mean(self.before)), 4),
            "mean_after": round(float(np.mean(self.after)), 4),
            "mean_difference": round(float(np.mean(self.before) - np.mean(self.after)), 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_before": len(self.before),
            "n_after": len(self.after),
        }

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """Execute full hypothesis-testing workflow."""
        hypotheses = self.state_hypotheses()
        normality = self.test_normality()
        test_result = self.select_and_run_test(normality)
        return {
            "hypotheses": hypotheses,
            "normality_tests": normality,
            "test_result": test_result,
        }


# ============================================================================
# 3. BIAS ASSESSMENT
# ============================================================================

class BiasAssessment:
    """
    Evaluate and disclose potential sources of bias:
    a) Geographic imbalance  (subreddit distribution as proxy)
    b) Platform algorithm bias  (qualitative disclosure)
    c) Time-window bias  (coverage gaps, weekday effects)
    d) Over-representation of high-activity users
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str = "created_utc",
        subreddit_column: str = "subreddit",
        author_column: str = "author_id_hash",
    ):
        self.df = df.copy()
        self.date_col = date_column
        self.sub_col = subreddit_column
        self.author_col = author_column
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

    # ------------------------------------------------------------------
    def geographic_imbalance(self) -> Dict:
        """
        Reddit has no explicit geolocation.  We use subreddit distribution
        as a proxy for community (and indirectly geographic) bias.
        Reports HHI (Herfindahl-Hirschman Index) and top subreddits.
        """
        if self.sub_col not in self.df.columns:
            return {"available": False, "note": "No subreddit column in data."}

        counts = self.df[self.sub_col].value_counts()
        shares = counts / counts.sum()
        hhi = float((shares ** 2).sum())

        return {
            "available": True,
            "n_unique_subreddits": int(counts.nunique()),
            "top_10_subreddits": counts.head(10).to_dict(),
            "hhi": round(hhi, 4),
            "hhi_interpretation": (
                "highly concentrated" if hhi > 0.25
                else "moderately concentrated" if hhi > 0.15
                else "low concentration"
            ),
            "concern": (
                "A small number of subreddits dominate the corpus. "
                "Results may not generalise beyond these communities."
                if hhi > 0.15
                else "Subreddit distribution shows reasonable diversity."
            ),
        }

    def platform_algorithm_bias(self) -> Dict:
        """
        Qualitative disclosure of Reddit-specific algorithmic biases.
        """
        return {
            "platform": "Reddit",
            "known_biases": [
                "Reddit's search API returns results ranked by 'relevance' or 'hot', "
                "biasing towards high-engagement content.",
                "Highly upvoted posts are over-represented in search results; "
                "low-score posts may be systematically excluded.",
                "Subreddit moderation policies may remove content, creating "
                "survivorship bias in the collected corpus.",
                "Reddit's user base skews younger, male, and US-centric "
                "(Pew Research, 2021), limiting demographic generalisability.",
            ],
            "mitigation_steps": [
                "Used multiple search queries across diverse subreddits.",
                "Collected data across a multi-year time span to reduce "
                "recency bias.",
                "Reported attrition at every preprocessing stage.",
            ],
        }

    def time_window_bias(self) -> Dict:
        """
        Examine temporal coverage: gaps, weekday concentration, and
        whether the chosen window introduces systematic bias.
        """
        dates = self.df[self.date_col].dt.date
        date_range = pd.date_range(dates.min(), dates.max(), freq="D")
        observed_dates = set(dates.unique())
        missing_dates = sorted(set(date_range.date) - observed_dates)

        # Weekday distribution
        weekday_counts = self.df[self.date_col].dt.day_name().value_counts()
        weekday_share = (weekday_counts / weekday_counts.sum()).to_dict()

        # Year distribution
        year_counts = self.df[self.date_col].dt.year.value_counts().sort_index()

        return {
            "date_range": {
                "start": str(dates.min()),
                "end": str(dates.max()),
                "span_days": (dates.max() - dates.min()).days,
            },
            "coverage": {
                "total_days_in_range": len(date_range),
                "days_with_data": len(observed_dates),
                "coverage_pct": round(
                    len(observed_dates) / max(1, len(date_range)) * 100, 1
                ),
                "missing_day_count": len(missing_dates),
            },
            "weekday_distribution": weekday_share,
            "year_distribution": year_counts.to_dict(),
            "concern": (
                "Significant temporal gaps detected; results may miss "
                "important events in uncovered periods."
                if len(missing_dates) / max(1, len(date_range)) > 0.30
                else "Temporal coverage is adequate."
            ),
        }

    def high_activity_user_bias(self) -> Dict:
        """
        Check if a small fraction of users account for a disproportionate
        share of posts (Lorenz-curve / Gini-style analysis).
        """
        if self.author_col not in self.df.columns:
            return {"available": False, "note": "No author column in data."}

        user_counts = self.df[self.author_col].value_counts()
        n_users = len(user_counts)
        n_posts = int(user_counts.sum())

        # Top 5 % of users
        top5_threshold = user_counts.quantile(0.95)
        top5_users = user_counts[user_counts >= top5_threshold]
        top5_post_share = float(top5_users.sum() / n_posts)

        # Gini coefficient
        sorted_counts = np.sort(user_counts.values).astype(float)
        cumulative = np.cumsum(sorted_counts)
        if len(sorted_counts) > 1:
            n_vals = len(sorted_counts)
            # Gini via trapezoidal integration (numpy ≥ 2 uses np.trapezoid)
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            if _trapz is not None:
                gini = float(1 - 2 * _trapz(cumulative / cumulative[-1], dx=1 / n_vals))
            else:
                # manual trapezoidal fallback
                y = cumulative / cumulative[-1]
                gini = float(1 - 2 * np.sum((y[:-1] + y[1:]) / 2) / n_vals)
        else:
            gini = 0.0

        return {
            "available": True,
            "n_unique_users": n_users,
            "total_posts": n_posts,
            "posts_per_user_mean": round(float(user_counts.mean()), 2),
            "posts_per_user_median": float(user_counts.median()),
            "posts_per_user_max": int(user_counts.max()),
            "top_5pct_threshold": float(top5_threshold),
            "top_5pct_user_count": int(len(top5_users)),
            "top_5pct_post_share": round(top5_post_share * 100, 2),
            "gini_coefficient": round(gini, 4),
            "concern": (
                f"Top 5 % of users contribute {top5_post_share*100:.1f} % of posts. "
                "High user concentration may bias aggregate sentiment."
                if top5_post_share > 0.30
                else "User activity distribution is reasonably balanced."
            ),
        }

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        return {
            "geographic_imbalance": self.geographic_imbalance(),
            "platform_algorithm_bias": self.platform_algorithm_bias(),
            "time_window_bias": self.time_window_bias(),
            "high_activity_user_bias": self.high_activity_user_bias(),
        }


# ============================================================================
# 4. SENSITIVITY ANALYSIS — TOP 5 % USERS
# ============================================================================

class SensitivityAnalysis:
    """
    1. Identify the top 5 % most-active users.
    2. Re-run the core analysis (sentiment distribution, mean score,
       t-test / Mann-Whitney) excluding those users.
    3. Compare and report differences.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        score_column: str = "sentiment_score",
        sentiment_column: str = "sentiment",
        date_column: str = "created_utc",
        author_column: str = "author_id_hash",
        policy_date: str = "2024-01-01",
    ):
        self.df = df.copy()
        self.score_col = score_column
        self.sent_col = sentiment_column
        self.date_col = date_column
        self.author_col = author_column
        self.policy_date = pd.to_datetime(policy_date)
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

    # helpers ---------------------------------------------------------------
    def _sentiment_summary(self, sub: pd.DataFrame) -> Dict:
        dist = sub[self.sent_col].value_counts()
        n = len(sub)
        return {
            "n": n,
            "positive_pct": round(dist.get("positive", 0) / max(1, n) * 100, 2),
            "neutral_pct": round(dist.get("neutral", 0) / max(1, n) * 100, 2),
            "negative_pct": round(dist.get("negative", 0) / max(1, n) * 100, 2),
            "mean_sentiment_score": round(float(sub[self.score_col].mean()), 4),
            "std_sentiment_score": round(float(sub[self.score_col].std()), 4),
        }

    def _run_test(self, sub: pd.DataFrame) -> Dict:
        before = sub.loc[sub[self.date_col] < self.policy_date, self.score_col].dropna().values
        after = sub.loc[sub[self.date_col] >= self.policy_date, self.score_col].dropna().values
        if len(before) < 2 or len(after) < 2:
            return {"error": "Insufficient data for test."}
        u, p = mannwhitneyu(before, after, alternative="two-sided")
        d = _cohens_d(before, after)
        ci_lo, ci_hi = _ci_mean_diff(before, after)
        return {
            "test": "Mann-Whitney U",
            "U": round(float(u), 2),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "cohens_d": round(d, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_before": len(before),
            "n_after": len(after),
        }

    # ------------------------------------------------------------------
    def identify_top_users(self, percentile: float = 0.95) -> Dict:
        if self.author_col not in self.df.columns:
            return {"available": False}
        user_counts = self.df[self.author_col].value_counts()
        threshold = user_counts.quantile(percentile)
        top_users = set(user_counts[user_counts >= threshold].index)
        return {
            "available": True,
            "percentile": percentile,
            "threshold_posts": float(threshold),
            "n_top_users": len(top_users),
            "n_total_users": len(user_counts),
            "posts_from_top_users": int(self.df[self.author_col].isin(top_users).sum()),
            "top_user_ids": list(top_users)[:20],   # sample for reference
        }

    def run(self) -> Dict:
        """Full sensitivity report: full vs. filtered."""
        top_info = self.identify_top_users()
        if not top_info.get("available", False):
            return {"available": False, "note": "No author column; cannot run user-based sensitivity."}

        top_user_ids = set(top_info.get("top_user_ids", []))
        # Re-derive full set (top_user_ids above is truncated to 20 for display)
        user_counts = self.df[self.author_col].value_counts()
        threshold = user_counts.quantile(0.95)
        all_top_users = set(user_counts[user_counts >= threshold].index)

        df_full = self.df
        df_filtered = self.df[~self.df[self.author_col].isin(all_top_users)]

        full_summary = self._sentiment_summary(df_full)
        filtered_summary = self._sentiment_summary(df_filtered)
        full_test = self._run_test(df_full)
        filtered_test = self._run_test(df_filtered)

        # Comparison
        diff_mean = round(
            full_summary["mean_sentiment_score"] - filtered_summary["mean_sentiment_score"], 4
        )
        conclusion_changed = (
            full_test.get("significant", None) != filtered_test.get("significant", None)
        )

        return {
            "top_users": top_info,
            "full_dataset": {
                "summary": full_summary,
                "hypothesis_test": full_test,
            },
            "filtered_dataset": {
                "summary": filtered_summary,
                "hypothesis_test": filtered_test,
            },
            "comparison": {
                "mean_score_difference": diff_mean,
                "conclusion_changed": conclusion_changed,
                "interpretation": (
                    "Excluding the top 5 % most-active users CHANGES the statistical "
                    "conclusion. Results are sensitive to high-activity users."
                    if conclusion_changed
                    else "Excluding the top 5 % most-active users does NOT change the "
                    "statistical conclusion. Results are robust to user concentration."
                ),
            },
        }


# ============================================================================
# 5. DRIVERS OF SENTIMENT — Logistic Regression & Feature Importance
# ============================================================================

class SentimentDriverAnalysis:
    """
    Identify and interpret the words that drive positive vs negative sentiment.

    Method:
    1. Build a binary label (positive=1 vs negative=0), excluding neutrals.
    2. Construct a term-frequency feature matrix (top-N unigrams).
    3. Fit a regularised logistic regression (sklearn Ridge / L2).
    4. Report odds ratios, coefficients, 95 % CI.
    5. Interpret top positive and negative drivers.

    Falls back to log-odds ratio analysis (no sklearn dependency) if sklearn
    is unavailable, preserving existing pipeline functionality.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "clean_text",
        sentiment_column: str = "sentiment",
        max_features: int = 500,
        min_df: int = 10,
    ):
        self.df = df.copy()
        self.text_col = text_column
        self.sent_col = sentiment_column
        self.max_features = max_features
        self.min_df = min_df

        # Filter to binary classification set
        self.df_binary = self.df[self.df[self.sent_col].isin(["positive", "negative"])].copy()
        self.df_binary["label"] = (self.df_binary[self.sent_col] == "positive").astype(int)

    # ------------------------------------------------------------------
    def _build_dtm(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build a simple document-term matrix (binary occurrence per doc).
        No sklearn dependency.
        """
        texts = self.df_binary[self.text_col].fillna("").str.lower()

        # Tokenize
        token_lists: List[List[str]] = []
        doc_freq: Counter = Counter()
        for text in texts:
            tokens = set(re.findall(r"\b[a-z]{3,}\b", text))
            token_lists.append(list(tokens))
            doc_freq.update(tokens)

        # Filter by min_df
        vocab = [
            w for w, c in doc_freq.most_common(self.max_features * 3)
            if c >= self.min_df
        ][: self.max_features]
        word2idx = {w: i for i, w in enumerate(vocab)}

        # Build matrix
        X = np.zeros((len(token_lists), len(vocab)), dtype=np.float32)
        for i, tokens in enumerate(token_lists):
            for t in tokens:
                if t in word2idx:
                    X[i, word2idx[t]] = 1.0

        return X, vocab

    def run_logistic_regression(self) -> Dict:
        """
        Fit logistic regression and extract odds ratios + CIs.
        Uses sklearn if available, otherwise falls back to log-odds.
        """
        if len(self.df_binary) < 50:
            return {"error": "Too few binary-labelled samples for regression."}

        X, vocab = self._build_dtm()
        y = self.df_binary["label"].values

        try:
            from sklearn.linear_model import LogisticRegression as LR
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler(with_mean=False)
            X_scaled = scaler.fit_transform(X)

            model = LR(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=1000, random_state=42,
            )
            model.fit(X_scaled, y)

            coefs = model.coef_[0]
            intercept = float(model.intercept_[0])

            # Odds ratios & approximate CIs (Wald)
            # SE from inverse Hessian approximation
            probs = model.predict_proba(X_scaled)[:, 1]
            W = np.diag(probs * (1 - probs))
            # For large matrices, use diagonal approximation
            diag_W = probs * (1 - probs)
            # Variance of beta_j ≈ 1 / sum(w_i * x_ij^2)
            se = np.zeros(len(coefs))
            for j in range(len(coefs)):
                denom = np.sum(diag_W * (X_scaled[:, j].toarray().ravel() if hasattr(X_scaled[:, j], 'toarray') else X_scaled[:, j]) ** 2)
                se[j] = 1.0 / np.sqrt(max(denom, 1e-10))

            z_crit = 1.96
            rows = []
            for j, word in enumerate(vocab):
                coef = float(coefs[j])
                odds_r = float(np.exp(coef))
                ci_lo = float(np.exp(coef - z_crit * se[j]))
                ci_hi = float(np.exp(coef + z_crit * se[j]))
                rows.append({
                    "word": word,
                    "coefficient": round(coef, 4),
                    "odds_ratio": round(odds_r, 4),
                    "or_ci_95_lower": round(ci_lo, 4),
                    "or_ci_95_upper": round(ci_hi, 4),
                    "significant": not (ci_lo <= 1.0 <= ci_hi),
                })

            results_df = pd.DataFrame(rows).sort_values("coefficient", ascending=False)

            # Accuracy
            acc = float((model.predict(X_scaled) == y).mean())

            positive_drivers = results_df.head(15).to_dict("records")
            negative_drivers = results_df.tail(15).iloc[::-1].to_dict("records")

            return {
                "method": "Logistic Regression (L2, sklearn)",
                "n_samples": len(y),
                "n_features": len(vocab),
                "intercept": round(intercept, 4),
                "accuracy": round(acc, 4),
                "positive_drivers": positive_drivers,
                "negative_drivers": negative_drivers,
                "interpretation": {
                    "how_identified": (
                        "A regularised (L2) logistic regression was fit to predict "
                        "positive (1) vs negative (0) sentiment from binary word-occurrence "
                        f"features ({len(vocab)} words, min document frequency = {self.min_df}). "
                        "Coefficients represent the change in log-odds of positive sentiment "
                        "associated with presence of each word."
                    ),
                    "odds_ratio_meaning": (
                        "An odds ratio > 1 means the word increases the odds of positive "
                        "sentiment; < 1 means it increases the odds of negative sentiment."
                    ),
                },
                "full_table": results_df.to_dict("records"),
            }

        except ImportError:
            return self._fallback_log_odds(X, y, vocab)

    def _fallback_log_odds(self, X: np.ndarray, y: np.ndarray, vocab: List[str]) -> Dict:
        """Fallback: log-odds ratio analysis (no sklearn needed)."""
        pos_mask = y == 1
        neg_mask = y == 0
        smoothing = 0.5

        rows = []
        for j, word in enumerate(vocab):
            pos_count = X[pos_mask, j].sum() + smoothing
            neg_count = X[neg_mask, j].sum() + smoothing
            total_pos = pos_mask.sum() + smoothing * 2
            total_neg = neg_mask.sum() + smoothing * 2

            p_pos = pos_count / total_pos
            p_neg = neg_count / total_neg
            odds_ratio = (p_pos / (1 - p_pos)) / (p_neg / (1 - p_neg))
            log_odds = float(np.log(odds_ratio))

            rows.append({
                "word": word,
                "log_odds": round(log_odds, 4),
                "odds_ratio": round(float(odds_ratio), 4),
                "count_positive": int(X[pos_mask, j].sum()),
                "count_negative": int(X[neg_mask, j].sum()),
            })

        results_df = pd.DataFrame(rows).sort_values("log_odds", ascending=False)
        return {
            "method": "Log-Odds Ratio (fallback, no sklearn)",
            "n_samples": len(y),
            "n_features": len(vocab),
            "positive_drivers": results_df.head(15).to_dict("records"),
            "negative_drivers": results_df.tail(15).iloc[::-1].to_dict("records"),
            "interpretation": {
                "how_identified": (
                    "Log-odds ratios were computed from binary word occurrence "
                    "in positive vs negative posts with Laplace smoothing."
                ),
                "odds_ratio_meaning": (
                    "An odds ratio > 1 means the word is more likely to appear "
                    "in positive-sentiment posts; < 1 in negative-sentiment posts."
                ),
            },
            "full_table": results_df.to_dict("records"),
        }

    # ------------------------------------------------------------------
    def plot_drivers(self, results: Dict, n: int = 15, save_path: Optional[str] = None) -> None:
        """Horizontal bar chart of top positive and negative drivers."""
        pos = results.get("positive_drivers", [])[:n]
        neg = results.get("negative_drivers", [])[:n]

        if not pos and not neg:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        # --- Positive drivers ---
        ax = axes[0]
        words = [d["word"] for d in pos]
        vals = [d.get("coefficient", d.get("log_odds", 0)) for d in pos]
        ors = [d.get("odds_ratio", 0) for d in pos]
        ax.barh(range(len(words)), vals, color="#6bcb77", edgecolor="#333", alpha=0.85)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        for i, (v, o) in enumerate(zip(vals, ors)):
            ax.text(v + 0.01, i, f"OR={o:.2f}", va="center", fontsize=8, alpha=0.7)
        ax.set_xlabel("Coefficient / Log-Odds", fontsize=11)
        ax.set_title("Positive Sentiment Drivers", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        # --- Negative drivers ---
        ax = axes[1]
        words = [d["word"] for d in neg]
        vals = [d.get("coefficient", d.get("log_odds", 0)) for d in neg]
        ors = [d.get("odds_ratio", 0) for d in neg]
        ax.barh(range(len(words)), vals, color="#ff6b6b", edgecolor="#333", alpha=0.85)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        for i, (v, o) in enumerate(zip(vals, ors)):
            ax.text(v - 0.01, i, f"OR={o:.2f}", va="center", ha="right", fontsize=8, alpha=0.7)
        ax.set_xlabel("Coefficient / Log-Odds", fontsize=11)
        ax.set_title("Negative Sentiment Drivers", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        plt.suptitle(
            "Drivers of Sentiment — Logistic Regression Feature Importance",
            fontsize=15, fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   Plot saved to: {save_path}")
        plt.close(fig)

    def run(self, save_dir: Optional[str] = None) -> Dict:
        """Run driver analysis end-to-end."""
        results = self.run_logistic_regression()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.plot_drivers(results, save_path=os.path.join(save_dir, "sentiment_drivers_regression.png"))
        return results


# ============================================================================
# COMBINED RUNNER
# ============================================================================

def run_statistical_analysis(
    df: pd.DataFrame,
    text_column: str = "clean_text",
    date_column: str = "created_utc",
    policy_date: str = "2024-01-01",
    output_dir: str = "results",
) -> Dict:
    """
    Master function: runs every checklist section and returns a single dict.

    Sections:
    1. data_quality          — duplicates, missing values, relevance, groups
    2. hypothesis_testing    — H₀/H₁, normality, test, effect size, CI
    3. bias_assessment       — geographic, platform, time-window, user
    4. sensitivity_analysis  — excluding top-5 % users, re-running tests
    5. sentiment_drivers     — logistic regression / log-odds, odds ratios
    """
    os.makedirs(output_dir, exist_ok=True)

    report: Dict[str, Any] = {}

    # ── 1. Data Quality ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STATISTICAL ANALYSIS — DATA & RELEVANCE")
    print("=" * 65)
    dq = DataQualityReport(df, text_column=text_column, date_column=date_column,
                           policy_date=policy_date)
    report["data_quality"] = dq.run()
    _q = report["data_quality"]
    print(f"  Duplicates:         {_q['duplicates']['exact_duplicates']} "
          f"({_q['duplicates']['duplicate_pct']} %)")
    print(f"  Missing cells:      {_q['missing_values']['total_missing_cells']}")
    print(f"  Relevance hit-rate: {_q['relevance']['keyword_hit_pct']} %")
    grp = _q["comparison_groups"]
    print(f"  Before group (< {grp['policy_date']}): n = {grp['before']['n']}")
    print(f"  After  group (≥ {grp['policy_date']}): n = {grp['after']['n']}")

    # ── 2. Hypothesis Testing ────────────────────────────────────────
    # Requires 'sentiment_score' column (set by BERT in Phase 3)
    if "sentiment_score" in df.columns:
        print("\n" + "=" * 65)
        print("  HYPOTHESIS & STATISTICAL TESTING")
        print("=" * 65)
        ht = HypothesisTestingSuite(df, score_column="sentiment_score",
                                    date_column=date_column, policy_date=policy_date)
        report["hypothesis_testing"] = ht.run()
        _ht = report["hypothesis_testing"]
        print(f"  H₀: {_ht['hypotheses']['H0']}")
        print(f"  H₁: {_ht['hypotheses']['H1']}")
        norm = _ht["normality_tests"]
        for g in ("before", "after"):
            if g in norm:
                sw = norm[g]
                print(f"  Shapiro-Wilk ({g}): W = {sw.get('shapiro_W','N/A')}, "
                      f"p = {sw.get('p_value','N/A')}, "
                      f"normal = {sw.get('normal_at_0_05','N/A')}")
        t = _ht["test_result"]
        print(f"  Test chosen:  {t.get('test_name','N/A')}")
        print(f"    {t.get('statistic_label','?')} = {t.get('statistic_value','N/A')}, "
              f"p = {t.get('p_value','N/A')}")
        print(f"    Cohen's d = {t.get('cohens_d','N/A')} ({t.get('cohens_d_interpretation','N/A')})")
        print(f"    95 % CI for mean diff: [{t.get('ci_95_lower','N/A')}, {t.get('ci_95_upper','N/A')}]")
        print(f"    Significant at α = .05: {t.get('significant_at_0_05','N/A')}")
    else:
        report["hypothesis_testing"] = {"skipped": True,
                                        "reason": "sentiment_score column not present — run BERT first."}
        print("\n  ⚠️  Hypothesis testing skipped (no sentiment_score column).")

    # ── 3. Bias Assessment ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BIAS ASSESSMENT")
    print("=" * 65)
    ba = BiasAssessment(df, date_column=date_column)
    report["bias_assessment"] = ba.run()
    _ba = report["bias_assessment"]
    geo = _ba["geographic_imbalance"]
    if geo.get("available"):
        print(f"  Subreddit HHI:         {geo['hhi']} ({geo['hhi_interpretation']})")
    plat = _ba["platform_algorithm_bias"]
    print(f"  Platform biases noted: {len(plat['known_biases'])} items")
    tw = _ba["time_window_bias"]
    print(f"  Temporal coverage:     {tw['coverage']['coverage_pct']} % of date range")
    ua = _ba["high_activity_user_bias"]
    if ua.get("available"):
        print(f"  Top 5 % user share:    {ua['top_5pct_post_share']} % of posts "
              f"(Gini = {ua['gini_coefficient']})")

    # ── 4. Sensitivity Analysis ──────────────────────────────────────
    if "sentiment_score" in df.columns and "sentiment" in df.columns:
        print("\n" + "=" * 65)
        print("  SENSITIVITY ANALYSIS — EXCLUDING TOP 5 % USERS")
        print("=" * 65)
        sa = SensitivityAnalysis(df, score_column="sentiment_score",
                                 sentiment_column="sentiment",
                                 date_column=date_column, policy_date=policy_date)
        report["sensitivity_analysis"] = sa.run()
        _sa = report["sensitivity_analysis"]
        if _sa.get("available", True) and "comparison" in _sa:
            fu = _sa["full_dataset"]["summary"]
            fi = _sa["filtered_dataset"]["summary"]
            print(f"  Full dataset:     n = {fu['n']}, mean = {fu['mean_sentiment_score']}")
            print(f"  Filtered dataset: n = {fi['n']}, mean = {fi['mean_sentiment_score']}")
            print(f"  Mean difference:  {_sa['comparison']['mean_score_difference']}")
            print(f"  Conclusion changed: {_sa['comparison']['conclusion_changed']}")
            print(f"  → {_sa['comparison']['interpretation']}")
        else:
            print("  ⚠️  Sensitivity analysis unavailable (no author column).")
    else:
        report["sensitivity_analysis"] = {"skipped": True,
                                          "reason": "sentiment columns not present."}
        print("\n  ⚠️  Sensitivity analysis skipped (no sentiment columns).")

    # ── 5. Drivers of Sentiment ──────────────────────────────────────
    if "sentiment" in df.columns:
        print("\n" + "=" * 65)
        print("  DRIVERS OF SENTIMENT")
        print("=" * 65)
        sd = SentimentDriverAnalysis(df, text_column=text_column,
                                     sentiment_column="sentiment")
        report["sentiment_drivers"] = sd.run(save_dir=output_dir)
        _sd = report["sentiment_drivers"]
        print(f"  Method: {_sd.get('method','N/A')}")
        print(f"  Samples: {_sd.get('n_samples','N/A')}, Features: {_sd.get('n_features','N/A')}")
        if "accuracy" in _sd:
            print(f"  Accuracy: {_sd['accuracy']}")
        print(f"\n  Top 5 POSITIVE drivers (increase odds of positive sentiment):")
        for d in _sd.get("positive_drivers", [])[:5]:
            print(f"    {d['word']:20s}  OR = {d.get('odds_ratio','N/A'):>7}  "
                  f"coef = {d.get('coefficient', d.get('log_odds','N/A'))}")
        print(f"\n  Top 5 NEGATIVE drivers (increase odds of negative sentiment):")
        for d in _sd.get("negative_drivers", [])[:5]:
            print(f"    {d['word']:20s}  OR = {d.get('odds_ratio','N/A'):>7}  "
                  f"coef = {d.get('coefficient', d.get('log_odds','N/A'))}")
        interp = _sd.get("interpretation", {})
        print(f"\n  How identified: {interp.get('how_identified','N/A')[:120]}...")
        print(f"  OR meaning:     {interp.get('odds_ratio_meaning','N/A')[:120]}")
    else:
        report["sentiment_drivers"] = {"skipped": True,
                                       "reason": "sentiment column not present."}
        print("\n  ⚠️  Sentiment driver analysis skipped (no sentiment column).")

    # ── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "statistical_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(_safe_serializable(report), f, indent=2)
    print(f"\n  Results saved → {out_path}")

    return report
