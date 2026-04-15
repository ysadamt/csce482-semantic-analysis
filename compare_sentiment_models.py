"""
RoBERTa vs VADER sentiment comparison utilities.

Generates:
- Top positive/negative posts for each model
- Agreement/disagreement visualizations
- A disagreement table with the most divergent posts
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LABEL_ORDER = ["negative", "neutral", "positive"]


def shorten_text(text: str, max_chars: int = 80) -> str:
    """Create compact labels suitable for plot axes."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def latest_analyzed_csv(results_dir: str = "results") -> str:
    """Return the newest analyzed_data CSV path."""
    candidates = sorted(glob.glob(os.path.join(results_dir, "analyzed_data_*.csv")))
    if not candidates:
        raise FileNotFoundError(
            f"No analyzed_data_*.csv files found under '{results_dir}'."
        )
    return candidates[-1]


def load_comparison_df(path: str, posts_only: bool = True) -> pd.DataFrame:
    """Load analyzed data and compute agreement diagnostics."""
    df = pd.read_csv(path)

    required = [
        "id",
        "raw_text",
        "sentiment",
        "sentiment_score",
        "vader_sentiment",
        "vader_sentiment_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )

    if posts_only and "content_type" in df.columns:
        content_type = df["content_type"].fillna("post").astype(str).str.lower()
        df = df[content_type == "post"].copy()

    # Keep only rows with valid text and scores for fair ranking.
    df["raw_text"] = df["raw_text"].fillna("").astype(str)
    df = df[df["raw_text"].str.strip() != ""].copy()
    df = df.dropna(subset=["sentiment_score", "vader_sentiment_score"]).copy()

    if len(df) == 0:
        raise ValueError("No usable rows found after filtering.")

    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    df["vader_sentiment"] = df["vader_sentiment"].astype(str).str.lower()
    df["score_diff_roberta_minus_vader"] = (
        df["sentiment_score"] - df["vader_sentiment_score"]
    )
    df["abs_score_diff"] = df["score_diff_roberta_minus_vader"].abs()
    df["label_agreement"] = df["sentiment"] == df["vader_sentiment"]
    return df


def select_top_posts(df: pd.DataFrame, score_col: str, top_n: int) -> Dict[str, pd.DataFrame]:
    """Get top positive and negative posts based on sentiment score."""
    top_positive = df.nlargest(top_n, score_col).copy()
    top_negative = df.nsmallest(top_n, score_col).copy()
    return {
        "positive": top_positive,
        "negative": top_negative,
    }


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach compact text columns for visual artifacts."""
    out = df.copy()
    out["text_preview"] = out["raw_text"].apply(lambda t: shorten_text(t, max_chars=140))
    return out


def save_top_tables(
    df: pd.DataFrame,
    output_dir: str,
    top_n: int,
) -> Dict[str, pd.DataFrame]:
    """Export top positive/negative posts for both models."""
    exports: Dict[str, pd.DataFrame] = {}

    roberta = select_top_posts(df, "sentiment_score", top_n)
    vader = select_top_posts(df, "vader_sentiment_score", top_n)

    for polarity, subset in roberta.items():
        key = f"top_{top_n}_roberta_{polarity}"
        table = add_display_columns(
            subset[
                [
                    "id",
                    "created_date",
                    "subreddit",
                    "sentiment",
                    "sentiment_score",
                    "vader_sentiment",
                    "vader_sentiment_score",
                    "abs_score_diff",
                    "raw_text",
                ]
            ]
        )
        table.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)
        exports[key] = table

    for polarity, subset in vader.items():
        key = f"top_{top_n}_vader_{polarity}"
        table = add_display_columns(
            subset[
                [
                    "id",
                    "created_date",
                    "subreddit",
                    "vader_sentiment",
                    "vader_sentiment_score",
                    "sentiment",
                    "sentiment_score",
                    "abs_score_diff",
                    "raw_text",
                ]
            ]
        )
        table.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)
        exports[key] = table

    return exports


def plot_score_scatter(df: pd.DataFrame, output_path: str) -> None:
    """Plot RoBERTa vs VADER score scatter with agreement highlighting."""
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df = df
    if len(plot_df) > 6000:
        plot_df = plot_df.sample(n=6000, random_state=42)

    palette = {True: "#2a9d8f", False: "#e76f51"}
    sns.scatterplot(
        data=plot_df,
        x="sentiment_score",
        y="vader_sentiment_score",
        hue="label_agreement",
        palette=palette,
        alpha=0.45,
        s=22,
        edgecolor=None,
        ax=ax,
    )

    ax.plot([-1, 1], [-1, 1], linestyle="--", color="black", linewidth=1)
    ax.set_title("RoBERTa vs VADER Sentiment Score (Actual Posts)", fontweight="bold")
    ax.set_xlabel("RoBERTa sentiment_score")
    ax.set_ylabel("VADER vader_sentiment_score")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(title="Label agreement")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_label_confusion(df: pd.DataFrame, output_path: str) -> None:
    """Plot confusion heatmap for RoBERTa labels vs VADER labels."""
    conf = pd.crosstab(df["sentiment"], df["vader_sentiment"])
    conf = conf.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title("Label Confusion: RoBERTa (rows) vs VADER (cols)", fontweight="bold")
    ax.set_xlabel("VADER label")
    ax.set_ylabel("RoBERTa label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_top_posts_grid(tables: Dict[str, pd.DataFrame], output_path: str, top_n: int) -> None:
    """Create a 2x2 grid showing top positive/negative posts by each model."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 14), constrained_layout=True)
    specs: List[tuple] = [
        ("top_{n}_roberta_positive", "sentiment_score", "Top RoBERTa Positive", "#2a9d8f"),
        ("top_{n}_roberta_negative", "sentiment_score", "Top RoBERTa Negative", "#264653"),
        (
            "top_{n}_vader_positive",
            "vader_sentiment_score",
            "Top VADER Positive",
            "#f4a261",
        ),
        (
            "top_{n}_vader_negative",
            "vader_sentiment_score",
            "Top VADER Negative",
            "#e76f51",
        ),
    ]

    for ax, (name_template, score_col, title, color) in zip(axes.flat, specs):
        key = name_template.format(n=top_n)
        subset = tables[key].copy()
        subset = subset.sort_values(score_col, ascending=True)
        labels = [shorten_text(t, max_chars=70) for t in subset["raw_text"]]

        y = np.arange(len(subset))
        ax.barh(y, subset[score_col], color=color, alpha=0.9)
        ax.set_yticks(y, labels)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(score_col)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle("Top Actual Posts by Sentiment Model", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def disagreement_table(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return posts rated differently, sorted by largest score gap."""
    disagree = df[df["sentiment"] != df["vader_sentiment"]].copy()
    if len(disagree) == 0:
        # If labels all agree, still show biggest score-distance rows.
        disagree = df.copy()

    cols = [
        "id",
        "created_date",
        "subreddit",
        "sentiment",
        "sentiment_score",
        "vader_sentiment",
        "vader_sentiment_score",
        "score_diff_roberta_minus_vader",
        "abs_score_diff",
        "raw_text",
    ]
    disagree = disagree.sort_values("abs_score_diff", ascending=False).head(top_n)
    disagree = disagree[cols].copy()
    disagree["text_preview"] = disagree["raw_text"].apply(
        lambda t: shorten_text(t, max_chars=140)
    )
    return disagree


def plot_disagreement_dumbbell(disagree_df: pd.DataFrame, output_path: str) -> None:
    """Visualize top disagreements with paired RoBERTa/VADER scores per post."""
    if len(disagree_df) == 0:
        return

    plot_df = disagree_df.copy().sort_values("abs_score_diff", ascending=True)
    labels = [shorten_text(t, max_chars=70) for t in plot_df["raw_text"]]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(14, max(8, len(plot_df) * 0.45)))
    for yi, rb, vd in zip(y, plot_df["sentiment_score"], plot_df["vader_sentiment_score"]):
        ax.plot([rb, vd], [yi, yi], color="#9aa0a6", linewidth=2, alpha=0.8)

    ax.scatter(
        plot_df["sentiment_score"],
        y,
        color="#1f77b4",
        s=55,
        label="RoBERTa",
        zorder=3,
    )
    ax.scatter(
        plot_df["vader_sentiment_score"],
        y,
        color="#ff7f0e",
        s=55,
        label="VADER",
        zorder=3,
    )

    ax.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Sentiment score")
    ax.set_title(
        "Posts Rated Differently: RoBERTa vs VADER (Top Score Gaps)",
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_comparison_artifacts(
    input_csv: str,
    output_dir: str,
    top_n: int,
    posts_only: bool,
) -> Dict[str, str]:
    """Main orchestration for all comparison outputs."""
    os.makedirs(output_dir, exist_ok=True)

    df = load_comparison_df(input_csv, posts_only=posts_only)
    tables = save_top_tables(df, output_dir=output_dir, top_n=top_n)

    scatter_path = os.path.join(output_dir, "roberta_vs_vader_score_scatter.png")
    plot_score_scatter(df, scatter_path)

    confusion_path = os.path.join(output_dir, "roberta_vs_vader_label_confusion.png")
    plot_label_confusion(df, confusion_path)

    top_grid_path = os.path.join(output_dir, f"top_{top_n}_posts_model_comparison.png")
    plot_top_posts_grid(tables, top_grid_path, top_n=top_n)

    disagree = disagreement_table(df, top_n=max(20, top_n * 4))
    disagree_csv_path = os.path.join(output_dir, "posts_rated_differently.csv")
    disagree.to_csv(disagree_csv_path, index=False)

    disagree_plot_path = os.path.join(output_dir, "posts_rated_differently_dumbbell.png")
    plot_disagreement_dumbbell(disagree, disagree_plot_path)

    summary = {
        "input_csv": input_csv,
        "rows_used": str(len(df)),
        "posts_only": str(posts_only),
        "label_agreement_rate": f"{df['label_agreement'].mean():.4f}",
        "mean_abs_score_diff": f"{df['abs_score_diff'].mean():.4f}",
        "score_scatter": scatter_path,
        "label_confusion": confusion_path,
        "top_posts_grid": top_grid_path,
        "differently_rated_posts_csv": disagree_csv_path,
        "differently_rated_posts_plot": disagree_plot_path,
    }

    for key in [
        f"top_{top_n}_roberta_positive",
        f"top_{top_n}_roberta_negative",
        f"top_{top_n}_vader_positive",
        f"top_{top_n}_vader_negative",
    ]:
        summary[f"{key}_csv"] = os.path.join(output_dir, f"{key}.csv")

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    summary["summary_csv"] = summary_path

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RoBERTa vs VADER post-level comparison outputs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Analyzed CSV path (default: latest results/analyzed_data_*.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison",
        help="Directory to write visualizations and tables",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top N posts for positive/negative tables (default: 5)",
    )
    parser.add_argument(
        "--include-comments",
        action="store_true",
        help="Include comment rows if content_type is present (default: posts only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = args.input or latest_analyzed_csv(results_dir="results")
    summary = generate_comparison_artifacts(
        input_csv=input_csv,
        output_dir=args.output_dir,
        top_n=max(1, int(args.top_n)),
        posts_only=not args.include_comments,
    )

    print("\nRoBERTa vs VADER comparison artifacts created:")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
