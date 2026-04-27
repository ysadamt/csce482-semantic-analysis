import unittest

import numpy as np
import pandas as pd

from analysis import build_weighted_sentiment_ensemble


class AnalysisEnsembleTests(unittest.TestCase):
    def _mock_scores(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sentiment": "positive",
                    "confidence": 0.92,
                    "prob_negative": 0.03,
                    "prob_neutral": 0.15,
                    "prob_positive": 0.82,
                    "climatebert_sentiment": "positive",
                    "climatebert_confidence": 0.88,
                    "climatebert_prob_negative": 0.04,
                    "climatebert_prob_neutral": 0.21,
                    "climatebert_prob_positive": 0.75,
                    "vader_sentiment": "positive",
                    "vader_confidence": 0.74,
                    "vader_prob_negative": 0.10,
                    "vader_prob_neutral": 0.20,
                    "vader_prob_positive": 0.70,
                },
                {
                    "sentiment": "negative",
                    "confidence": 0.86,
                    "prob_negative": 0.79,
                    "prob_neutral": 0.16,
                    "prob_positive": 0.05,
                    "climatebert_sentiment": "negative",
                    "climatebert_confidence": 0.81,
                    "climatebert_prob_negative": 0.72,
                    "climatebert_prob_neutral": 0.21,
                    "climatebert_prob_positive": 0.07,
                    "vader_sentiment": "neutral",
                    "vader_confidence": 0.67,
                    "vader_prob_negative": 0.31,
                    "vader_prob_neutral": 0.59,
                    "vader_prob_positive": 0.10,
                },
            ]
        )

    def test_ensemble_uses_all_three_models(self):
        df = self._mock_scores()

        out, details = build_weighted_sentiment_ensemble(df)

        self.assertEqual(len(out), 2)
        self.assertEqual(details["models_used"], ["roberta", "climatebert", "vader"])
        self.assertIn("ensemble_sentiment", out.columns)
        self.assertIn("ensemble_sentiment_score", out.columns)
        self.assertIn("agreement_with_roberta", details)
        self.assertIn("agreement_with_climatebert", details)
        self.assertIn("agreement_with_vader", details)

        prob_sum = (
            out["ensemble_prob_negative"]
            + out["ensemble_prob_neutral"]
            + out["ensemble_prob_positive"]
        )
        self.assertTrue(np.allclose(prob_sum.values, np.ones(len(out)), atol=1e-6))
        self.assertEqual(out["ensemble_sentiment"].iloc[0], "positive")
        self.assertEqual(out["ensemble_sentiment"].iloc[1], "negative")

    def test_ensemble_falls_back_to_two_models(self):
        df = self._mock_scores().drop(
            columns=[
                "climatebert_sentiment",
                "climatebert_confidence",
                "climatebert_prob_negative",
                "climatebert_prob_neutral",
                "climatebert_prob_positive",
            ]
        )

        out, details = build_weighted_sentiment_ensemble(df)

        self.assertEqual(details["models_used"], ["roberta", "vader"])
        self.assertIn("ensemble_sentiment", out.columns)
        self.assertEqual(out["ensemble_sentiment"].iloc[0], "positive")

    def test_ensemble_requires_at_least_two_models(self):
        df = self._mock_scores().drop(
            columns=[
                "climatebert_sentiment",
                "climatebert_confidence",
                "climatebert_prob_negative",
                "climatebert_prob_neutral",
                "climatebert_prob_positive",
                "vader_sentiment",
                "vader_confidence",
                "vader_prob_negative",
                "vader_prob_neutral",
                "vader_prob_positive",
            ]
        )

        with self.assertRaises(ValueError):
            build_weighted_sentiment_ensemble(df)


if __name__ == "__main__":
    unittest.main(verbosity=2)
