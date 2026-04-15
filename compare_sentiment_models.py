"""
RoBERTa vs ClimateBERT sentiment comparison utilities.

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
import torch
from scipy.special import softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


LABEL_ORDER = ["negative", "neutral", "positive"]
DEFAULT_CLIMATEBERT_MODEL = "climatebert/distilroberta-base-climate-sentiment"


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


def _normalize_label_name(label: str) -> str | None:
    """Map model-specific labels to canonical sentiment labels."""
    text = str(label).strip().lower()
    if any(token in text for token in ["positive", "opportunity", "optimistic", "bull"]):
        return "positive"
    if any(token in text for token in ["negative", "risk", "pessimistic", "bear"]):
        return "negative"
    if any(token in text for token in ["neutral", "mixed", "none"]):
        return "neutral"
    return None


class ClimateBERTAnalyzer:
    """ClimateBERT sentiment inference wrapper."""

    def __init__(self, model_name: str = DEFAULT_CLIMATEBERT_MODEL):
        print(f"Loading ClimateBERT model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.label_map = self._build_label_map()
        print("✅ ClimateBERT model loaded")

    def _build_label_map(self) -> Dict[int, str]:
        """Build index->sentiment mapping from config labels with safe fallback."""
        id2label = getattr(self.config, "id2label", None) or {}

        mapped: Dict[int, str] = {}
        normalized_labels: Dict[int, str] = {}

        for key, value in id2label.items():
            try:
                idx = int(key)
            except Exception:
                continue
            normalized_labels[idx] = str(value)

        if not normalized_labels and getattr(self.config, "num_labels", 0) > 0:
            for idx in range(int(self.config.num_labels)):
                normalized_labels[idx] = f"label_{idx}"

        for idx, raw_label in normalized_labels.items():
            guessed = _normalize_label_name(raw_label)
            if guessed is not None:
                mapped[idx] = guessed

        default_order = {0: "negative", 1: "neutral", 2: "positive"}
        for idx in sorted(normalized_labels.keys()):
            mapped.setdefault(idx, default_order.get(idx, "neutral"))

        return mapped

    def _preprocess(self, text: str) -> str:
        words = []
        for word in str(text).split():
            if word.startswith("@"):
                words.append("@user")
            elif word.startswith("http"):
                words.append("http")
            else:
                words.append(word)
        return " ".join(words)

    def predict_sentiment(self, texts: List[str], batch_size: int = 32) -> pd.DataFrame:
        """Run ClimateBERT inference and emit standardized sentiment columns."""
        rows = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_processed = [self._preprocess(t)[:512] for t in batch]
                encoded = self.tokenizer(
                    batch_processed,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                output = self.model(**encoded)
                probs = softmax(output.logits.detach().numpy(), axis=1)

                for prob_row in probs:
                    probs_by_sentiment = {
                        "negative": 0.0,
                        "neutral": 0.0,
                        "positive": 0.0,
                    }
                    for idx, value in enumerate(prob_row):
                        sentiment = self.label_map.get(idx, "neutral")
                        if sentiment not in probs_by_sentiment:
                            sentiment = "neutral"
                        probs_by_sentiment[sentiment] += float(value)

                    pred_sentiment = max(
                        probs_by_sentiment,
                        key=lambda label: probs_by_sentiment[label],
                    )

                    rows.append(
                        {
                            "climatebert_sentiment": pred_sentiment,
                            "climatebert_confidence": float(probs_by_sentiment[pred_sentiment]),
                            "climatebert_prob_negative": float(probs_by_sentiment["negative"]),
                            "climatebert_prob_neutral": float(probs_by_sentiment["neutral"]),
                            "climatebert_prob_positive": float(probs_by_sentiment["positive"]),
                            "climatebert_sentiment_score": float(
                                probs_by_sentiment["positive"] - probs_by_sentiment["negative"]
                            ),
                        }
                    )

                if (i // batch_size) % 5 == 0:
                    print(
                        f"   ClimateBERT processed {min(i + batch_size, len(texts))}/{len(texts)} rows..."
                    )

        return pd.DataFrame(rows)


def _attach_climatebert_predictions(
    df: pd.DataFrame,
    run_climatebert: bool,
    climatebert_model_name: str,
    batch_size: int,
) -> pd.DataFrame:
    """Add ClimateBERT columns either by reuse (if present) or by inference."""
    required_climatebert = [
        "climatebert_sentiment",
        "climatebert_sentiment_score",
    ]
    has_required = all(col in df.columns for col in required_climatebert)

    if has_required and not run_climatebert:
        return df

    if has_required and run_climatebert:
        print("\nRecomputing ClimateBERT predictions for all selected rows...")
    elif not run_climatebert:
        raise ValueError(
            "ClimateBERT columns are missing in input CSV. "
            "Run without --reuse-existing-climatebert to compute them."
        )

    analyzer = ClimateBERTAnalyzer(model_name=climatebert_model_name)
    climatebert_df = analyzer.predict_sentiment(
        df["raw_text"].fillna("").astype(str).tolist(),
        batch_size=batch_size,
    )

    for col in climatebert_df.columns:
        df[col] = climatebert_df[col].values

    return df


def load_comparison_df(
    path: str,
    posts_only: bool = True,
    run_climatebert: bool = True,
    climatebert_model_name: str = DEFAULT_CLIMATEBERT_MODEL,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Load analyzed data and compute RoBERTa-vs-ClimateBERT diagnostics."""
    df = pd.read_csv(path)

    required = [
        "id",
        "raw_text",
        "sentiment",
        "sentiment_score",
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
    df = df.dropna(subset=["sentiment_score"]).copy()

    if len(df) == 0:
        raise ValueError("No usable rows found after filtering.")

    df = _attach_climatebert_predictions(
        df,
        run_climatebert=run_climatebert,
        climatebert_model_name=climatebert_model_name,
        batch_size=batch_size,
    )
    df = df.dropna(subset=["climatebert_sentiment_score"]).copy()

    if len(df) == 0:
        raise ValueError("No usable rows found after filtering.")

    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    df["climatebert_sentiment"] = df["climatebert_sentiment"].astype(str).str.lower()
    df["score_diff_roberta_minus_climatebert"] = (
        df["sentiment_score"] - df["climatebert_sentiment_score"]
    )
    df["abs_score_diff"] = df["score_diff_roberta_minus_climatebert"].abs()
    df["label_agreement"] = df["sentiment"] == df["climatebert_sentiment"]
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
    climatebert = select_top_posts(df, "climatebert_sentiment_score", top_n)

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
                    "climatebert_sentiment",
                    "climatebert_sentiment_score",
                    "abs_score_diff",
                    "raw_text",
                ]
            ]
        )
        table.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)
        exports[key] = table

    for polarity, subset in climatebert.items():
        key = f"top_{top_n}_climatebert_{polarity}"
        table = add_display_columns(
            subset[
                [
                    "id",
                    "created_date",
                    "subreddit",
                    "climatebert_sentiment",
                    "climatebert_sentiment_score",
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
    """Plot RoBERTa vs ClimateBERT score scatter with agreement highlighting."""
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df = df
    if len(plot_df) > 6000:
        plot_df = plot_df.sample(n=6000, random_state=42)

    palette = {True: "#2a9d8f", False: "#e76f51"}
    sns.scatterplot(
        data=plot_df,
        x="sentiment_score",
        y="climatebert_sentiment_score",
        hue="label_agreement",
        palette=palette,
        alpha=0.45,
        s=22,
        edgecolor=None,
        ax=ax,
    )

    ax.plot([-1, 1], [-1, 1], linestyle="--", color="black", linewidth=1)
    ax.set_title("RoBERTa vs ClimateBERT Sentiment Score (Actual Posts)", fontweight="bold")
    ax.set_xlabel("RoBERTa sentiment_score")
    ax.set_ylabel("ClimateBERT climatebert_sentiment_score")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(title="Label agreement")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_label_confusion(df: pd.DataFrame, output_path: str) -> None:
    """Plot confusion heatmap for RoBERTa labels vs ClimateBERT labels."""
    conf = pd.crosstab(df["sentiment"], df["climatebert_sentiment"])
    conf = conf.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title("Label Confusion: RoBERTa (rows) vs ClimateBERT (cols)", fontweight="bold")
    ax.set_xlabel("ClimateBERT label")
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
            "top_{n}_climatebert_positive",
            "climatebert_sentiment_score",
            "Top ClimateBERT Positive",
            "#f4a261",
        ),
        (
            "top_{n}_climatebert_negative",
            "climatebert_sentiment_score",
            "Top ClimateBERT Negative",
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
    disagree = df[df["sentiment"] != df["climatebert_sentiment"]].copy()
    if len(disagree) == 0:
        # If labels all agree, still show biggest score-distance rows.
        disagree = df.copy()

    cols = [
        "id",
        "created_date",
        "subreddit",
        "sentiment",
        "sentiment_score",
        "climatebert_sentiment",
        "climatebert_sentiment_score",
        "score_diff_roberta_minus_climatebert",
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
    """Visualize top disagreements with paired RoBERTa/ClimateBERT scores per post."""
    if len(disagree_df) == 0:
        return

    plot_df = disagree_df.copy().sort_values("abs_score_diff", ascending=True)
    labels = [shorten_text(t, max_chars=70) for t in plot_df["raw_text"]]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(14, max(8, len(plot_df) * 0.45)))
    for yi, rb, cb in zip(y, plot_df["sentiment_score"], plot_df["climatebert_sentiment_score"]):
        ax.plot([rb, cb], [yi, yi], color="#9aa0a6", linewidth=2, alpha=0.8)

    ax.scatter(
        plot_df["sentiment_score"],
        y,
        color="#1f77b4",
        s=55,
        label="RoBERTa",
        zorder=3,
    )
    ax.scatter(
        plot_df["climatebert_sentiment_score"],
        y,
        color="#ff7f0e",
        s=55,
        label="ClimateBERT",
        zorder=3,
    )

    ax.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Sentiment score")
    ax.set_title(
        "Posts Rated Differently: RoBERTa vs ClimateBERT (Top Score Gaps)",
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
    run_climatebert: bool = True,
    climatebert_model_name: str = DEFAULT_CLIMATEBERT_MODEL,
    batch_size: int = 32,
) -> Dict[str, str]:
    """Main orchestration for all comparison outputs."""
    os.makedirs(output_dir, exist_ok=True)

    df = load_comparison_df(
        input_csv,
        posts_only=posts_only,
        run_climatebert=run_climatebert,
        climatebert_model_name=climatebert_model_name,
        batch_size=batch_size,
    )

    enriched_input_path = os.path.join(output_dir, "comparison_input_with_climatebert.csv")
    df.to_csv(enriched_input_path, index=False)

    tables = save_top_tables(df, output_dir=output_dir, top_n=top_n)

    scatter_path = os.path.join(output_dir, "roberta_vs_climatebert_score_scatter.png")
    plot_score_scatter(df, scatter_path)

    confusion_path = os.path.join(output_dir, "roberta_vs_climatebert_label_confusion.png")
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
        "comparison_input_with_climatebert": enriched_input_path,
        "climatebert_model": climatebert_model_name,
        "climatebert_inference_ran": str(run_climatebert),
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
        f"top_{top_n}_climatebert_positive",
        f"top_{top_n}_climatebert_negative",
    ]:
        summary[f"{key}_csv"] = os.path.join(output_dir, f"{key}.csv")

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    summary["summary_csv"] = summary_path

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RoBERTa vs ClimateBERT post-level comparison outputs."
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
        "--climatebert-model",
        type=str,
        default=DEFAULT_CLIMATEBERT_MODEL,
        help="Hugging Face model ID for climate sentiment comparison",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for ClimateBERT (default: 32)",
    )
    parser.add_argument(
        "--reuse-existing-climatebert",
        action="store_true",
        help="Reuse existing climatebert_* columns if present instead of rerunning inference",
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
        run_climatebert=not args.reuse_existing_climatebert,
        climatebert_model_name=args.climatebert_model,
        batch_size=max(1, int(args.batch_size)),
    )

    print("\nRoBERTa vs ClimateBERT comparison artifacts created:")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
