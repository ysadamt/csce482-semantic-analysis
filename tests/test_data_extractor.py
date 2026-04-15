import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from data_extractor import RedditDataExtractor, create_methane_dairy_queries


class FakeComment:
    def __init__(self, body: str):
        self.body = body


class FakeSubredditRef:
    def __init__(self, name: str):
        self.display_name = name


class FakePost:
    def __init__(
        self,
        post_id: str,
        title: str,
        body: str,
        created_utc: int,
        subreddit: str,
        score: int = 10,
        num_comments: int = 0,
        comments=None,
    ):
        self.id = post_id
        self.title = title
        self.selftext = body
        self.created_utc = created_utc
        self.score = score
        self.upvote_ratio = 0.9
        self.num_comments = num_comments
        self.subreddit = FakeSubredditRef(subreddit)
        self.is_self = True
        self.is_video = False
        self.permalink = f"/r/{subreddit}/comments/{post_id}/sample"
        self.author = "unit_test_user"
        self.comment_sort = None
        self.comment_limit = None
        self.comments = [FakeComment(text) for text in (comments or [])]


class FakeSubredditSearch:
    def __init__(self, posts):
        self._posts = posts

    def search(self, query, sort=None, time_filter=None, limit=None):
        limit = len(self._posts) if limit is None else int(limit)
        return iter(self._posts[:limit])


class FakeReddit:
    def __init__(self, posts):
        self._posts = list(posts)
        self._index = {p.id: p for p in self._posts}

    def subreddit(self, _subreddit_name):
        return FakeSubredditSearch(self._posts)

    def info(self, fullnames=None):
        return []

    def submission(self, id=None):
        return self._index[str(id)]


class DataExtractorTests(unittest.TestCase):
    def test_create_methane_dairy_queries_are_unique(self):
        queries = create_methane_dairy_queries()
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(any("dairy" in q.lower() for q in queries))
        self.assertTrue(any("methane" in q.lower() for q in queries))

    def test_extract_posts_counts_posts_plus_comments_toward_target(self):
        now_ts = int(datetime(2024, 2, 1).timestamp())
        fake_posts = [
            FakePost(
                post_id="p1",
                title="Dairy methane solutions",
                body="Seaweed can reduce enteric methane.",
                created_utc=now_ts,
                subreddit="climate",
                num_comments=2,
                comments=["Great idea", "Needs policy support"],
            ),
            FakePost(
                post_id="p2",
                title="Livestock climate concern",
                body="Emissions remain high.",
                created_utc=now_ts,
                subreddit="climate",
                num_comments=0,
            ),
            FakePost(
                post_id="p3",
                title="Farm methane capture",
                body="Digesters are scaling.",
                created_utc=now_ts,
                subreddit="climate",
                num_comments=0,
            ),
        ]

        fake_reddit = FakeReddit(fake_posts)
        with patch("data_extractor.praw.Reddit", return_value=fake_reddit):
            extractor = RedditDataExtractor(client_id="x", client_secret="y")

        df = extractor.extract_posts(
            queries=["dairy methane"],
            subreddits=["climate"],
            target_count=3,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2025, 1, 1),
            sort_methods=["relevance"],
            time_filters=["all"],
            include_comments=True,
            max_comments_per_post=5,
            fast_mode=True,
            temporal_balance=False,
            pool_multiplier=1,
        )

        self.assertEqual(len(df), 3)
        self.assertEqual(int((df["content_type"] == "post").sum()), 1)
        self.assertEqual(int((df["content_type"] == "comment").sum()), 2)
        self.assertEqual(extractor.attrition_log.get("N1_total_records"), 3)
        self.assertEqual(extractor.attrition_log.get("N1_unique_posts"), 1)
        self.assertEqual(extractor.attrition_log.get("N1_comment_rows"), 2)

    def test_enrich_comments_skips_non_post_rows(self):
        fake_reddit = FakeReddit([])
        with patch("data_extractor.praw.Reddit", return_value=fake_reddit):
            extractor = RedditDataExtractor(client_id="x", client_secret="y")

        in_df = pd.DataFrame(
            [
                {
                    "id": "p10",
                    "content_type": "post",
                    "num_comments": 0,
                    "comments_text": "",
                    "comments_collected_count": 0,
                },
                {
                    "id": "p10_c1",
                    "content_type": "comment",
                    "num_comments": 0,
                    "comments_text": "",
                    "comments_collected_count": 0,
                },
            ]
        )

        out_df = extractor.enrich_comments_for_dataframe(
            in_df,
            max_comments_per_post=5,
            progress_every=0,
        )

        self.assertEqual(len(out_df), 2)
        self.assertEqual(extractor.attrition_log.get("comment_enrichment_attempted_posts"), 0)
        self.assertEqual(extractor.attrition_log.get("comment_enrichment_skipped_non_post_rows"), 1)
        self.assertEqual(extractor.attrition_log.get("comment_enrichment_skipped_zero_comment_posts"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
