import os
import tempfile
import unittest

import matplotlib
import pandas as pd

from compare_sentiment_models import (
    disagreement_table,
    generate_comparison_artifacts,
    load_comparison_df,
)


matplotlib.use("Agg")


class CompareSentimentModelsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.input_csv = os.path.join(self.tmp.name, "analyzed.csv")

        rows = [
            {
                "id": "p1",
                "raw_text": "Great climate progress and hopeful news.",
                "sentiment": "positive",
                "sentiment_score": 0.9,
                "vader_sentiment": "positive",
                "vader_sentiment_score": 0.7,
                "created_date": "2024-01-01",
                "subreddit": "climate",
                "content_type": "post",
            },
            {
                "id": "p2",
                "raw_text": "This policy is terrible and harmful.",
                "sentiment": "negative",
                "sentiment_score": -0.8,
                "vader_sentiment": "positive",
                "vader_sentiment_score": 0.4,
                "created_date": "2024-01-02",
                "subreddit": "environment",
                "content_type": "post",
            },
            {
                "id": "p3",
                "raw_text": "Mixed evidence but worth discussing.",
                "sentiment": "neutral",
                "sentiment_score": 0.05,
                "vader_sentiment": "negative",
                "vader_sentiment_score": -0.4,
                "created_date": "2024-01-03",
                "subreddit": "science",
                "content_type": "post",
            },
            {
                "id": "p4",
                "raw_text": "Methane emissions are still concerning.",
                "sentiment": "negative",
                "sentiment_score": -0.3,
                "vader_sentiment": "negative",
                "vader_sentiment_score": -0.2,
                "created_date": "2024-01-04",
                "subreddit": "farming",
                "content_type": "post",
            },
            {
                "id": "p5",
                "raw_text": "Strong innovation momentum in clean tech.",
                "sentiment": "positive",
                "sentiment_score": 0.6,
                "vader_sentiment": "neutral",
                "vader_sentiment_score": 0.0,
                "created_date": "2024-01-05",
                "subreddit": "energy",
                "content_type": "post",
            },
            {
                "id": "p5_c1",
                "raw_text": "I agree with this post.",
                "sentiment": "positive",
                "sentiment_score": 0.95,
                "vader_sentiment": "positive",
                "vader_sentiment_score": 0.99,
                "created_date": "2024-01-05",
                "subreddit": "energy",
                "content_type": "comment",
            },
        ]
        pd.DataFrame(rows).to_csv(self.input_csv, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_comparison_df_filters_comments_and_adds_fields(self):
        df = load_comparison_df(self.input_csv, posts_only=True)
        self.assertEqual(len(df), 5)
        self.assertIn("abs_score_diff", df.columns)
        self.assertIn("label_agreement", df.columns)
        self.assertTrue((df["content_type"] == "post").all())

    def test_disagreement_table_prefers_label_disagreements(self):
        df = load_comparison_df(self.input_csv, posts_only=True)
        table = disagreement_table(df, top_n=3)

        self.assertEqual(len(table), 3)
        self.assertTrue((table["sentiment"] != table["vader_sentiment"]).all())
        self.assertGreaterEqual(table["abs_score_diff"].iloc[0], table["abs_score_diff"].iloc[-1])

    def test_generate_comparison_artifacts_creates_expected_outputs(self):
        output_dir = os.path.join(self.tmp.name, "comparison_out")
        summary = generate_comparison_artifacts(
            input_csv=self.input_csv,
            output_dir=output_dir,
            top_n=2,
            posts_only=True,
        )

        required_paths = [
            summary["score_scatter"],
            summary["label_confusion"],
            summary["top_posts_grid"],
            summary["differently_rated_posts_csv"],
            summary["differently_rated_posts_plot"],
            summary["summary_csv"],
            summary["top_2_roberta_positive_csv"],
            summary["top_2_roberta_negative_csv"],
            summary["top_2_vader_positive_csv"],
            summary["top_2_vader_negative_csv"],
        ]
        for path in required_paths:
            self.assertTrue(os.path.exists(path), f"Missing output: {path}")

        self.assertEqual(summary["rows_used"], "5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
