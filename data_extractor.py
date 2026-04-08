"""
Phase 1: Reddit Data Extraction
A Reproducible Reddit API Workflow for Peer-Reviewed Research

This module provides transparent, auditable, and policy-relevant social media corpus
collection suitable for hypothesis-driven analysis and long-term reproducibility.
"""

import praw
import pandas as pd
import numpy as np
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import time
import re

load_dotenv()


WHITESPACE_RE = re.compile(r'\s+')


class RedditDataExtractor:
    """
    Reddit data extraction with full audit trail and reproducibility features.
    
    Features:
    - Query logging for reproducibility
    - Pagination support
    - Deduplication
    - User ID anonymization (SHA-256 hashing)
    - Attrition tracking
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None, 
                 user_agent: str = "research_sentiment_analyzer_v2"):
        """Initialize the Reddit API client."""
        self.reddit = praw.Reddit(
            client_id=client_id or os.getenv("REDDIT_KEY"),
            client_secret=client_secret or os.getenv("REDDIT_SECRET"),
            user_agent=user_agent
        )
        self.query_log = []
        self.attrition_log = {}
        
    def _hash_user_id(self, user_id: str) -> str:
        """Anonymize user IDs using SHA-256 hashing for IRB compliance."""
        return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]
    
    def _log_query(self, query: str, subreddit: str, sort: str, 
                   time_filter: str, results_count: int):
        """Log query parameters for reproducibility."""
        self.query_log.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "subreddit": subreddit,
            "sort": sort,
            "time_filter": time_filter,
            "results_count": results_count,
            "api_tier": "standard"
        })

    def _is_livestock_methane_relevant(self, text: str) -> bool:
        """Require both livestock and methane/emissions cues to reduce off-topic results."""
        text = str(text).lower()
        livestock_terms = {
            'dairy', 'livestock', 'cattle', 'cow', 'cows', 'beef', 'ruminant',
            'enteric', 'farm', 'farming', 'herd', 'manure'
        }
        methane_terms = {
            'methane', 'emission', 'emissions', 'ghg', 'greenhouse gas',
            'greenhouse', 'climate', 'enteric fermentation'
        }
        has_livestock = any(term in text for term in livestock_terms)
        has_methane = any(term in text for term in methane_terms)
        return has_livestock and has_methane

    def _collect_comments(self, post, max_comments: int) -> Tuple[List[str], bool]:
        """Collect up to max_comments comment bodies, skipping placeholders."""
        if max_comments <= 0:
            return [], False

        if getattr(post, 'num_comments', 0) <= 0:
            return [], False

        comments = []
        try:
            # Prefer top-level comments to minimize API overhead during enrichment.
            post.comment_sort = 'top'
            post.comment_limit = max_comments
            for comment in post.comments:
                if len(comments) >= max_comments:
                    break

                body = getattr(comment, 'body', '')
                if not body:
                    continue

                body = WHITESPACE_RE.sub(' ', str(body)).strip()
                if not body:
                    continue

                if body.lower() in {'[deleted]', '[removed]'}:
                    continue

                comments.append(body)

            return comments, False
        except Exception:
            return [], True

    def enrich_comments_for_dataframe(self,
                                      df: pd.DataFrame,
                                      id_column: str = 'id',
                                      content_type_column: str = 'content_type',
                                      comments_column: str = 'comments_text',
                                      comments_count_column: str = 'comments_collected_count',
                                      max_comments_per_post: int = 10,
                                      skip_existing_comments: bool = True,
                                      progress_every: int = 250,
                                      batch_size: int = 100) -> pd.DataFrame:
        """Fetch comments for existing post rows using post IDs (no post rescraping)."""
        if id_column not in df.columns:
            raise ValueError(
                f"Input data must contain '{id_column}' column to fetch comments by submission ID"
            )

        if len(df) == 0:
            df_out = df.copy()
            if comments_column not in df_out.columns:
                df_out[comments_column] = ''
            if comments_count_column not in df_out.columns:
                df_out[comments_count_column] = 0
            return df_out

        df_out = df.copy()
        if comments_column not in df_out.columns:
            df_out[comments_column] = ''
        if comments_count_column not in df_out.columns:
            df_out[comments_count_column] = 0

        max_comments_per_post = max(0, int(max_comments_per_post))

        if max_comments_per_post == 0:
            self.attrition_log['comment_enrichment_source'] = 'existing_csv'
            self.attrition_log['comment_enrichment_rows'] = int(len(df_out))
            self.attrition_log['comment_enrichment_max_per_post'] = 0
            self.attrition_log['comment_enrichment_attempted_posts'] = 0
            self.attrition_log['comment_enrichment_failed_posts'] = 0
            self.attrition_log['comment_enrichment_total_comments'] = 0
            self.attrition_log['comment_enrichment_skipped_existing'] = 0
            self.attrition_log['comment_enrichment_skipped_non_post_rows'] = 0
            self.attrition_log['comment_enrichment_skipped_zero_comment_posts'] = 0
            self.attrition_log['comment_enrichment_avg_comments_per_attempted_post'] = 0.0
            return df_out

        attempted = 0
        failed = 0
        total_comments = 0
        skipped_existing = 0
        skipped_non_post_rows = 0
        skipped_zero_comment_posts = 0

        print("\nCollecting comments for existing posts (no rescrape)...")

        # Build an explicit worklist so we can batch submission lookups.
        targets = []
        for row_num, (idx, row) in enumerate(df_out.iterrows(), start=1):
            if content_type_column in df_out.columns:
                content_type = str(row.get(content_type_column, 'post')).strip().lower()
                if content_type and content_type != 'post':
                    skipped_non_post_rows += 1
                    df_out.at[idx, comments_column] = ''
                    df_out.at[idx, comments_count_column] = 0
                    continue

            post_id = str(row[id_column]).strip()
            existing_text = row.get(comments_column, '')
            has_existing = isinstance(existing_text, str) and existing_text.strip() != ''

            if skip_existing_comments and has_existing:
                skipped_existing += 1
                continue

            existing_num_comments = row.get('num_comments', None)
            if pd.notna(existing_num_comments):
                try:
                    if int(existing_num_comments) <= 0:
                        skipped_zero_comment_posts += 1
                        df_out.at[idx, comments_column] = ''
                        df_out.at[idx, comments_count_column] = 0
                        continue
                except Exception:
                    pass

            targets.append((row_num, idx, post_id))

        attempted = len(targets)

        batch_size = max(1, min(int(batch_size), 100))
        processed_targets = 0
        for batch_start in range(0, len(targets), batch_size):
            batch = targets[batch_start: batch_start + batch_size]
            fullnames = [f"t3_{post_id}" for _, _, post_id in batch]
            submissions_by_id = {}

            try:
                for submission in self.reddit.info(fullnames=fullnames):
                    submissions_by_id[str(getattr(submission, 'id', ''))] = submission
            except Exception:
                # Fall back to per-ID retrieval for this batch if bulk lookup fails.
                submissions_by_id = {}

            for row_num, idx, post_id in batch:
                try:
                    submission = submissions_by_id.get(post_id)
                    if submission is None:
                        submission = self.reddit.submission(id=post_id)

                    comments, fetch_failed = self._collect_comments(
                        submission,
                        max_comments=max_comments_per_post
                    )

                    if fetch_failed:
                        failed += 1

                    total_comments += len(comments)
                    df_out.at[idx, comments_column] = " || ".join(comments)
                    df_out.at[idx, comments_count_column] = len(comments)
                except Exception:
                    failed += 1
                    df_out.at[idx, comments_column] = ''
                    df_out.at[idx, comments_count_column] = 0

                processed_targets += 1
                if progress_every > 0 and processed_targets % progress_every == 0:
                    print(
                        f"  comments processed for {processed_targets}/{attempted} attempted posts"
                    )

        if attempted > 0 and progress_every > 0 and processed_targets % progress_every != 0:
            print(f"  comments processed for {processed_targets}/{attempted} attempted posts")

        self.attrition_log['comment_enrichment_source'] = 'existing_csv'
        self.attrition_log['comment_enrichment_rows'] = int(len(df_out))
        self.attrition_log['comment_enrichment_max_per_post'] = int(max_comments_per_post)
        self.attrition_log['comment_enrichment_attempted_posts'] = int(attempted)
        self.attrition_log['comment_enrichment_failed_posts'] = int(failed)
        self.attrition_log['comment_enrichment_total_comments'] = int(total_comments)
        self.attrition_log['comment_enrichment_skipped_existing'] = int(skipped_existing)
        self.attrition_log['comment_enrichment_skipped_non_post_rows'] = int(
            skipped_non_post_rows
        )
        self.attrition_log['comment_enrichment_skipped_zero_comment_posts'] = int(
            skipped_zero_comment_posts
        )
        self.attrition_log['comment_enrichment_avg_comments_per_attempted_post'] = (
            float(total_comments / attempted) if attempted > 0 else 0.0
        )

        return df_out
    
    def construct_queries(self, primary_concepts: List[str], 
                         contextual_constraints: List[str] = None,
                         exclude_terms: List[str] = None) -> List[str]:
        """
        Construct Boolean logic queries following best practices:
        (Primary Concept A OR Primary Concept B) AND (Contextual Constraint)
        
        Args:
            primary_concepts: Main topic terms with OR relationship
            contextual_constraints: Domain relevance terms (AND relationship)
            exclude_terms: Terms to exclude from results
        """
        queries = []
        
        # Base query from primary concepts
        base_query = " OR ".join(primary_concepts)
        
        if contextual_constraints:
            for constraint in contextual_constraints:
                query = f"({base_query}) {constraint}"
                queries.append(query)
        else:
            queries.append(base_query)
        
        # Also create individual queries for better coverage
        for concept in primary_concepts:
            if contextual_constraints:
                for constraint in contextual_constraints:
                    queries.append(f"{concept} {constraint}")
            else:
                queries.append(concept)
        
        return list(set(queries))
    
    def extract_posts(self, 
                     queries: List[str],
                     subreddits: List[str] = None,
                     target_count: int = 3000,
                     start_date: datetime = None,
                     end_date: datetime = None,
                     sort_methods: List[str] = None,
                     time_filters: List[str] = None,
                     include_comments: bool = False,
                     max_comments_per_post: int = 10,
                     fast_mode: bool = False,
                     temporal_balance: bool = True,
                     pool_multiplier: int = 2,
                     random_seed: int = 42) -> pd.DataFrame:
        """
        Extract posts from Reddit with full metadata collection.
        
        Args:
            queries: List of search queries
            subreddits: List of subreddits to search (default: ['all'])
            target_count: Target number of output rows
            start_date: Filter posts from this date onwards
            end_date: Filter posts until this date
            sort_methods: Reddit sort methods
            time_filters: Time filter options
            include_comments: Whether to emit comments as separate rows
            max_comments_per_post: Maximum number of comments emitted per post
            fast_mode: Reduce search breadth for quicker extraction
        
        Returns:
            DataFrame with posts and metadata
        """
        # Only scrape from subreddits directly relevant to
        # methane / dairy / livestock / climate discourse.
        # 'all' is intentionally excluded to avoid off-topic noise.
        subreddits = subreddits or [
            # Climate & environment
            'climate', 'environment', 'climatechange', 'globalwarming',
            'ClimateActionPlan', 'ClimateOffensive', 'sustainability',
            'zerowaste', 'renewableenergy',
            # Agriculture & livestock
            'farming', 'agriculture', 'ranching', 'homesteading',
            'dairy', 'Cattle',
            # Science & policy
            'science', 'EverythingScience', 'environmental_science',
            'energy', 'green',
            # Food systems & ethics
            'vegan', 'vegetarian', 'AnimalRights',
            'foodscience', 'Futurology',
        ]
        if fast_mode:
            sort_methods = sort_methods or ['relevance', 'new']
            time_filters = time_filters or ['year', 'all']
            pool_multiplier = 1
        else:
            sort_methods = sort_methods or ['relevance', 'top', 'comments', 'new']
            time_filters = time_filters or ['all', 'year']
        start_date = start_date or datetime(2018, 1, 1)
        end_date = end_date or datetime.now()
        candidate_target = min(
            max(target_count, target_count * max(1, pool_multiplier)),
            target_count + (400 if fast_mode else 1500),
        )
        max_comments_per_post = max(0, int(max_comments_per_post))
        
        seen_ids = set()
        data = []
        post_cache = {}
        comment_fetch_attempted_posts = 0
        comment_fetch_failed_posts = 0
        comment_fetch_total_comments = 0
        
        print(f"\n{'='*60}")
        print("REDDIT DATA EXTRACTION - PHASE 1")
        print(f"{'='*60}")
        print(f"Target: {target_count} output rows (posts + comments)")
        print(f"Candidate pool target: {candidate_target} posts")
        print(f"Fast mode: {'ON' if fast_mode else 'OFF'}")
        if include_comments and max_comments_per_post > 0:
            print("Row mode: comments are emitted as rows and count toward target")
        else:
            print("Row mode: posts only")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Queries: {len(queries)}")
        print(f"Subreddits: {', '.join(subreddits)}")
        print(f"{'='*60}\n")
        
        # Track initial attempt count
        total_attempts = 0

        def run_search_pass(search_queries: List[str],
                            search_subreddits: List[str],
                            search_sorts: List[str],
                            search_time_filters: List[str],
                            pass_name: str,
                            enforce_relevance: bool = False) -> None:
            nonlocal total_attempts, data, seen_ids

            print(f"\n--- {pass_name} ---")
            for query in search_queries:
                if len(data) >= candidate_target:
                    break

                for subreddit in search_subreddits:
                    if len(data) >= candidate_target:
                        break

                    for sort in search_sorts:
                        if len(data) >= candidate_target:
                            break

                        for time_filter in search_time_filters:
                            if len(data) >= candidate_target:
                                break

                            try:
                                results_this_query = 0
                                remaining_needed = candidate_target - len(data)
                                if fast_mode:
                                    search_limit = min(50, max(15, remaining_needed))
                                else:
                                    search_limit = min(100, max(25, remaining_needed))

                                for post in self.reddit.subreddit(subreddit).search(
                                    query, sort=sort, time_filter=time_filter, limit=search_limit
                                ):
                                    total_attempts += 1

                                    post_date = datetime.fromtimestamp(post.created_utc)
                                    if post_date < start_date or post_date > end_date:
                                        continue

                                    if post.id in seen_ids:
                                        continue

                                    full_text = f"{post.title} {post.selftext}".strip()
                                    if enforce_relevance and not self._is_livestock_methane_relevant(full_text):
                                        continue

                                    seen_ids.add(post.id)
                                    results_this_query += 1

                                    post_data = {
                                        'id': post.id,
                                        'raw_text': full_text,
                                        'title': post.title,
                                        'body': post.selftext,
                                        'created_utc': post_date,
                                        'created_date': post_date.strftime('%Y-%m-%d'),
                                        'created_year': post_date.year,
                                        'created_month': post_date.month,
                                        'score': post.score,
                                        'upvote_ratio': post.upvote_ratio,
                                        'num_comments': post.num_comments,
                                        'engagement_total': post.score + post.num_comments,
                                        'subreddit': post.subreddit.display_name,
                                        'is_self': post.is_self,
                                        'is_video': post.is_video,
                                        'permalink': f"https://reddit.com{post.permalink}",
                                        'author_id_hash': self._hash_user_id(
                                            str(post.author) if post.author else 'deleted'
                                        ),
                                        'search_query': query,
                                        'search_subreddit': subreddit,
                                        'query_pass': pass_name,
                                        'comments_text': '',
                                        'comments_collected_count': 0,
                                    }

                                    # Reuse submission objects later to avoid one lookup call per selected post.
                                    post_cache[post.id] = post

                                    data.append(post_data)

                                    if len(data) >= candidate_target:
                                        break

                                self._log_query(query, subreddit, sort, time_filter, results_this_query)

                                if results_this_query > 0:
                                    print(
                                        f"  [{len(data):>5}/{candidate_target}] "
                                        f"q='{query[:30]}...' r/{subreddit} "
                                        f"sort={sort} t={time_filter} -> +{results_this_query}"
                                    )

                            except Exception as e:
                                if "rate" in str(e).lower():
                                    print("  Rate limited, waiting 60s...")
                                    time.sleep(60)
                                continue
        
        # Pass 1: strict query strategy over curated topical subreddits.
        run_search_pass(
            search_queries=queries,
            search_subreddits=subreddits,
            search_sorts=sort_methods,
            search_time_filters=time_filters,
            pass_name='strict_subreddit_pass',
            enforce_relevance=False,
        )

        # Pass 2: adaptive fallback to fill the candidate pool target.
        if len(data) < candidate_target:
            remaining = candidate_target - len(data)
            print(
                f"\nUnder candidate pool target after strict pass ({len(data)}/{candidate_target}). "
                f"Launching adaptive fallback for ~{remaining} additional posts."
            )
            fallback_queries = [
                'methane dairy',
                'methane livestock',
                'cattle methane emissions',
                'dairy greenhouse gas',
                'enteric fermentation cattle',
                'livestock emissions climate',
                'dairy methane reduction',
                'manure methane farm',
                'beef methane emissions',
                'ruminant methane',
                'agricultural methane',
            ]
            run_search_pass(
                search_queries=fallback_queries,
                search_subreddits=['all'],
                search_sorts=['relevance', 'new'] if fast_mode else ['relevance', 'new', 'top', 'comments'],
                search_time_filters=['year', 'month'] if fast_mode else ['all', 'year', 'month', 'week'],
                pass_name='adaptive_fallback_all',
                enforce_relevance=True,
            )

        # Pass 3: broad fallback to maximize sample fill if still under candidate target.
        if len(data) < candidate_target and not fast_mode:
            remaining = candidate_target - len(data)
            print(
                f"\nStill under candidate target after adaptive fallback ({len(data)}/{candidate_target}). "
                f"Launching broad fallback for ~{remaining} additional posts."
            )
            broad_queries = [
                'methane',
                'livestock',
                'dairy',
                'cattle',
                'emissions',
                'greenhouse gas',
                'agriculture climate',
                'enteric fermentation',
                'farm emissions',
                'manure methane',
                'beef climate',
                'cow burps methane',
                'dairy sustainability',
                'livestock ghg',
            ]
            run_search_pass(
                search_queries=broad_queries,
                search_subreddits=['all'],
                search_sorts=['relevance', 'new', 'top', 'comments'],
                search_time_filters=['all', 'year', 'month', 'week'],
                pass_name='broad_fallback_all',
                enforce_relevance=False,
            )

        # Pass 4: combinational query expansion to push toward full candidate pool.
        if len(data) < candidate_target and not fast_mode:
            remaining = candidate_target - len(data)
            print(
                f"\nStill under candidate target after broad fallback ({len(data)}/{candidate_target}). "
                f"Launching combinational expansion for ~{remaining} additional posts."
            )
            livestock_terms = [
                'dairy', 'livestock', 'cattle', 'beef', 'cow', 'ruminant', 'farm', 'manure'
            ]
            methane_terms = [
                'methane', 'emissions', 'greenhouse gas', 'ghg', 'climate', 'enteric fermentation'
            ]
            expansion_queries = [
                f"{livestock} {methane}"
                for livestock in livestock_terms
                for methane in methane_terms
            ]
            run_search_pass(
                search_queries=expansion_queries,
                search_subreddits=['all'],
                search_sorts=['relevance', 'new', 'comments', 'top'],
                search_time_filters=['all', 'year', 'month', 'week', 'day'],
                pass_name='combinational_fallback_all',
                enforce_relevance=False,
            )

        self.attrition_log['extraction_target_count'] = int(target_count)
        self.attrition_log['candidate_pool_target'] = int(candidate_target)
        self.attrition_log['fast_mode_enabled'] = bool(fast_mode)
        self.attrition_log['candidate_pool_collected_before_balancing'] = int(len(data))
        self.attrition_log['strict_pass_unique_posts'] = int(
            sum(1 for row in data if row.get('query_pass') == 'strict_subreddit_pass')
        )
        self.attrition_log['adaptive_fallback_unique_posts'] = int(
            sum(1 for row in data if row.get('query_pass') == 'adaptive_fallback_all')
        )
        self.attrition_log['broad_fallback_unique_posts'] = int(
            sum(1 for row in data if row.get('query_pass') == 'broad_fallback_all')
        )
        self.attrition_log['combinational_fallback_unique_posts'] = int(
            sum(1 for row in data if row.get('query_pass') == 'combinational_fallback_all')
        )
        
        # Create DataFrame
        df = pd.DataFrame(data)

        # Temporal balancing to avoid recency skew
        if len(df) > 0 and temporal_balance:
            before_year_dist = df['created_year'].value_counts().sort_index().to_dict()
            df = self._balance_by_year(
                df=df,
                target_count=target_count,
                start_date=start_date,
                end_date=end_date,
                random_seed=random_seed
            )
            after_year_dist = df['created_year'].value_counts().sort_index().to_dict()
            self.attrition_log['temporal_balance_before'] = before_year_dist
            self.attrition_log['temporal_balance_after'] = after_year_dist
            self.attrition_log['temporal_balance_enabled'] = True

        # Assemble final rows. Comments are emitted as their own rows and
        # count toward the same target_count budget as posts.
        final_rows = []
        post_rows_added = 0
        comment_rows_added = 0

        if len(df) > 0:
            if include_comments and max_comments_per_post > 0:
                print("\nCollecting comments and assembling post/comment rows...")
            else:
                print("\nAssembling post rows...")

            for idx, post_row in enumerate(df.to_dict('records'), start=1):
                if len(final_rows) >= target_count:
                    break

                post_id = str(post_row.get('id', ''))

                post_record = dict(post_row)
                post_record['content_type'] = 'post'
                post_record['parent_post_id'] = post_id
                post_record['comments_text'] = ''
                post_record['comments_collected_count'] = 0
                final_rows.append(post_record)
                post_rows_added += 1

                if not include_comments or max_comments_per_post <= 0:
                    continue

                num_comments_value = post_row.get('num_comments', 0)
                try:
                    if int(num_comments_value) <= 0:
                        continue
                except Exception:
                    continue

                comment_fetch_attempted_posts += 1
                try:
                    submission = post_cache.get(post_id)
                    if submission is None:
                        submission = self.reddit.submission(id=post_id)

                    comments, fetch_failed = self._collect_comments(
                        submission,
                        max_comments=max_comments_per_post
                    )
                    if fetch_failed:
                        comment_fetch_failed_posts += 1

                    comment_fetch_total_comments += len(comments)
                    post_record['comments_collected_count'] = len(comments)

                    for comment_idx, comment_text in enumerate(comments, start=1):
                        if len(final_rows) >= target_count:
                            break

                        comment_record = dict(post_row)
                        comment_record['id'] = f"{post_id}_c{comment_idx}"
                        comment_record['raw_text'] = comment_text
                        comment_record['title'] = ''
                        comment_record['body'] = comment_text
                        comment_record['score'] = 0
                        comment_record['upvote_ratio'] = np.nan
                        comment_record['num_comments'] = 0
                        comment_record['engagement_total'] = 0
                        comment_record['permalink'] = (
                            f"{post_row.get('permalink', '')}#comment-{comment_idx}"
                        )
                        comment_record['author_id_hash'] = self._hash_user_id(
                            f"comment_{post_id}_{comment_idx}"
                        )
                        comment_record['comments_text'] = ''
                        comment_record['comments_collected_count'] = 0
                        comment_record['content_type'] = 'comment'
                        comment_record['parent_post_id'] = post_id

                        final_rows.append(comment_record)
                        comment_rows_added += 1
                except Exception:
                    comment_fetch_failed_posts += 1

                if idx % 500 == 0:
                    print(
                        f"  assembled {len(final_rows)}/{target_count} rows "
                        f"from {idx}/{len(df)} candidate posts"
                    )

        if len(final_rows) > 0:
            df = pd.DataFrame(final_rows)
        else:
            df = df.head(0).copy()
            if 'content_type' not in df.columns:
                df['content_type'] = pd.Series(dtype='object')
            if 'parent_post_id' not in df.columns:
                df['parent_post_id'] = pd.Series(dtype='object')

        self.attrition_log['comment_fetch_enabled'] = bool(include_comments)
        self.attrition_log['comment_fetch_max_per_post'] = int(max_comments_per_post)
        self.attrition_log['comment_fetch_attempted_posts'] = int(comment_fetch_attempted_posts)
        self.attrition_log['comment_fetch_failed_posts'] = int(comment_fetch_failed_posts)
        self.attrition_log['comment_fetch_total_comments'] = int(comment_fetch_total_comments)
        self.attrition_log['comment_fetch_avg_comments_per_attempted_post'] = (
            float(comment_fetch_total_comments / comment_fetch_attempted_posts)
            if comment_fetch_attempted_posts > 0 else 0.0
        )
        self.attrition_log['comment_rows_added'] = int(comment_rows_added)
        
        # Record attrition from extraction phase
        self.attrition_log['N0_raw_extracted'] = total_attempts
        self.attrition_log['N1_after_date_filter'] = int(len(df))
        self.attrition_log['N1_total_records'] = int(len(df))
        self.attrition_log['N1_unique_posts'] = int(post_rows_added)
        self.attrition_log['N1_comment_rows'] = int(comment_rows_added)
        
        print(f"\n{'='*60}")
        print(f"✅ EXTRACTION COMPLETE")
        print(f"   Total API hits: {total_attempts}")
        print(f"   Total rows collected: {len(df)}")
        print(f"   Post rows: {post_rows_added}")
        print(f"   Comment rows: {comment_rows_added}")
        print(f"   Date range achieved: {df['created_date'].min()} to {df['created_date'].max()}" 
              if len(df) > 0 else "   No posts collected")
        if len(df) > 0:
            year_dist = df['created_year'].value_counts().sort_index()
            print("   Year distribution (balanced):")
            print(year_dist.to_string())
        print(f"{'='*60}\n")
        
        return df

    def _balance_by_year(self,
                         df: pd.DataFrame,
                         target_count: int,
                         start_date: datetime,
                         end_date: datetime,
                         random_seed: int = 42) -> pd.DataFrame:
        """Sample approximately equal counts per year from start_date to end_date."""
        if len(df) <= target_count:
            return df

        years = list(range(start_date.year, end_date.year + 1))
        if not years:
            return df.sample(n=target_count, random_state=random_seed)

        base_quota = target_count // len(years)
        remainder = target_count % len(years)

        parts = []
        used_idx = set()

        for i, year in enumerate(years):
            year_df = df[df['created_year'] == year]
            if len(year_df) == 0:
                continue
            quota = base_quota + (1 if i < remainder else 0)
            take = min(quota, len(year_df))
            sampled = year_df.sample(n=take, random_state=random_seed + year)
            parts.append(sampled)
            used_idx.update(sampled.index.tolist())

        if parts:
            balanced = pd.concat(parts, axis=0)
        else:
            balanced = df.sample(n=min(target_count, len(df)), random_state=random_seed)

        # Fill any shortfall from remaining pool while preserving date constraints
        shortfall = target_count - len(balanced)
        if shortfall > 0:
            remaining = df.drop(index=list(used_idx), errors='ignore')
            if len(remaining) > 0:
                extra_take = min(shortfall, len(remaining))
                extra = remaining.sample(n=extra_take, random_state=random_seed + 999)
                balanced = pd.concat([balanced, extra], axis=0)

        # If still larger than target_count (edge-case), trim deterministically
        if len(balanced) > target_count:
            balanced = balanced.sample(n=target_count, random_state=random_seed)

        return balanced.sort_values('created_utc').reset_index(drop=True)
    
    def manual_relevance_audit(self, df: pd.DataFrame, 
                               sample_size: int = 100) -> Tuple[pd.DataFrame, float]:
        """
        Perform manual relevance audit of first N posts.
        Returns sample and estimated relevance percentage.
        
        Note: In production, this should be done manually by researchers.
        Here we provide the sample for review.
        """
        sample = df.head(sample_size)[['raw_text', 'subreddit', 'score']].copy()
        sample['is_relevant'] = None  # To be filled manually
        
        print(f"\n📋 MANUAL RELEVANCE AUDIT")
        print(f"   Please review {sample_size} posts and mark relevance")
        print(f"   Target: ≥80% relevance rate")
        
        return sample
    
    def save_data(self, df: pd.DataFrame, 
                  output_dir: str = "data",
                  format: str = "parquet",
                  prefix: str = "reddit_extraction") -> Dict[str, str]:
        """
        Save extracted data with full documentation.
        
        Args:
            df: DataFrame to save
            output_dir: Output directory
            format: 'parquet' (recommended) or 'csv'
            prefix: Filename prefix
            
        Returns:
            Dictionary with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        paths = {}
        
        # Save main data
        if format == "parquet":
            data_path = os.path.join(output_dir, f"{prefix}_{timestamp}.parquet")
            df.to_parquet(data_path, index=False)
        else:
            data_path = os.path.join(output_dir, f"{prefix}_{timestamp}.csv")
            df.to_csv(data_path, index=False)
        paths['data'] = data_path
        
        # Save query log (for reproducibility)
        query_log_path = os.path.join(output_dir, f"{prefix}_query_log_{timestamp}.json")
        with open(query_log_path, 'w') as f:
            json.dump(self.query_log, f, indent=2)
        paths['query_log'] = query_log_path
        
        # Save attrition log
        attrition_path = os.path.join(output_dir, f"{prefix}_attrition_{timestamp}.json")
        with open(attrition_path, 'w') as f:
            json.dump(self.attrition_log, f, indent=2)
        paths['attrition_log'] = attrition_path

        if 'content_type' in df.columns and len(df) > 0:
            content_types = df['content_type'].fillna('post').astype(str).str.lower()
            total_post_rows = int((content_types == 'post').sum())
            total_comment_rows = int((content_types == 'comment').sum())
        else:
            total_post_rows = int(len(df))
            total_comment_rows = 0
        
        # Generate extraction summary
        summary = {
            "extraction_date": timestamp,
            "total_records": int(len(df)),
            "total_posts": total_post_rows,
            "total_comment_rows": total_comment_rows,
            "date_range": {
                "start": df['created_date'].min() if len(df) > 0 else None,
                "end": df['created_date'].max() if len(df) > 0 else None
            },
            "subreddits": df['subreddit'].value_counts().to_dict() if len(df) > 0 else {},
            "queries_executed": len(self.query_log),
            "attrition": self.attrition_log,
            "files": paths
        }
        
        summary_path = os.path.join(output_dir, f"{prefix}_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        paths['summary'] = summary_path
        
        print(f"\n💾 DATA SAVED:")
        for key, path in paths.items():
            print(f"   {key}: {path}")
        
        return paths
    
    def get_attrition_report(self) -> pd.DataFrame:
        """Generate attrition report table for publication."""
        report = pd.DataFrame([
            {"Stage": stage, "Description": stage.replace("_", " ").title(), "Count": count}
            for stage, count in self.attrition_log.items()
        ])
        return report


def create_methane_dairy_queries() -> List[str]:
    """
    Create multiple Boolean query strategies for methane/livestock discourse.

    These are intentionally structured to improve construct validity by requiring
    livestock-related context with methane/emissions language, while reducing
    off-topic hits (e.g., automotive/industrial emissions).
    """
    queries = [
        # Core boolean strategy requested in project guidance
        '("dairy" OR "livestock") AND ("methane" OR "emission" OR "emissions")',

        # Expand livestock terms while keeping methane/emissions anchored
        '("dairy" OR "livestock" OR "cattle" OR "cow" OR "beef") AND ("methane" OR "greenhouse gas" OR "ghg")',

        # Enteric fermentation pathway (directly livestock methane source)
        '("enteric fermentation" OR "ruminant") AND ("methane" OR "emissions")',

        # Dairy production and climate framing
        '("dairy farming" OR "dairy industry") AND ("methane" OR "climate" OR "emissions")',

        # Mitigation and technology discourse
        '("feed additive" OR "methane digester" OR "anaerobic digester") AND ("cattle" OR "dairy" OR "livestock")',

        # Policy framing for agricultural methane
        '("agriculture" OR "livestock") AND ("methane policy" OR "methane regulation" OR "methane target")',

        # Sustainability-oriented livestock discussions
        '("regenerative" OR "sustainable") AND ("dairy" OR "livestock") AND ("methane" OR "emissions")',

        # Explicit exclusion to reduce non-livestock emission topics
        '("dairy" OR "livestock" OR "cattle") AND ("methane" OR "emissions") NOT ("car" OR "vehicle" OR "factory" OR "industrial")',

        # Plain-language methane framing often used in public discourse
        '("cow burps" OR "cows") AND ("methane" OR "climate change")',
    ]

    # Preserve insertion order while removing accidental duplicates.
    return list(dict.fromkeys(queries))


def extract_reddit_data(target_count: int = 3000,
                        start_year: int = 2018,
                        include_comments: bool = True,
                        max_comments_per_post: int = 10,
                        fast_mode: bool = False) -> pd.DataFrame:
    """
    Convenience function to run full extraction pipeline.
    
    Args:
        target_count: Target output rows (posts + comments, 2500-3000 recommended)
        start_year: Start year for data collection (2018 default)
        include_comments: Include comments in extraction output
        max_comments_per_post: Maximum comments captured per post
        fast_mode: Reduce extraction search breadth for faster runtime
    
    Returns:
        DataFrame with extracted posts
    """
    extractor = RedditDataExtractor()
    
    # Get predefined multi-strategy Boolean queries for methane/dairy research.
    queries = create_methane_dairy_queries()
    
    print(f"📝 Generated {len(queries)} query variations")
    
    # Extract posts
    df = extractor.extract_posts(
        queries=queries,
        target_count=target_count,
        start_date=datetime(start_year, 1, 1),
        include_comments=include_comments,
        max_comments_per_post=max_comments_per_post,
        fast_mode=fast_mode,
    )
    
    # Save data
    if len(df) > 0:
        extractor.save_data(df, output_dir="data", format="csv")
    
    return df, extractor


if __name__ == "__main__":
    # Run extraction
    df, extractor = extract_reddit_data(
        target_count=3000,
        start_year=2018,
        include_comments=True,
        max_comments_per_post=10,
        fast_mode=False,
    )
    
    # Print summary
    if len(df) > 0:
        if 'content_type' in df.columns:
            content_types = df['content_type'].fillna('post').astype(str).str.lower()
            total_post_rows = int((content_types == 'post').sum())
            total_comment_rows = int((content_types == 'comment').sum())
        else:
            total_post_rows = int(len(df))
            total_comment_rows = 0

        print("\n📊 EXTRACTION SUMMARY:")
        print(f"   Total rows: {len(df)}")
        print(f"   Post rows: {total_post_rows}")
        print(f"   Comment rows: {total_comment_rows}")
        print(f"   Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        print(f"   Unique subreddits: {df['subreddit'].nunique()}")
        print(f"\n   Top subreddits:")
        print(df['subreddit'].value_counts().head(10).to_string())
        print(f"\n   Posts per year:")
        print(df['created_year'].value_counts().sort_index().to_string())
