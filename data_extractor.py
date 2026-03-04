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
                     temporal_balance: bool = True,
                     pool_multiplier: int = 4,
                     random_seed: int = 42) -> pd.DataFrame:
        """
        Extract posts from Reddit with full metadata collection.
        
        Args:
            queries: List of search queries
            subreddits: List of subreddits to search (default: ['all'])
            target_count: Target number of unique posts
            start_date: Filter posts from this date onwards
            end_date: Filter posts until this date
            sort_methods: Reddit sort methods
            time_filters: Time filter options
            include_comments: Whether to include top comments
        
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
        sort_methods = sort_methods or ['relevance', 'top', 'comments', 'new']
        time_filters = time_filters or ['all', 'year']
        start_date = start_date or datetime(2018, 1, 1)
        end_date = end_date or datetime.now()
        candidate_target = max(target_count, target_count * max(1, pool_multiplier))
        
        seen_ids = set()
        data = []
        
        print(f"\n{'='*60}")
        print("REDDIT DATA EXTRACTION - PHASE 1")
        print(f"{'='*60}")
        print(f"Target: {target_count} unique posts")
        print(f"Candidate pool target: {candidate_target} posts")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Queries: {len(queries)}")
        print(f"Subreddits: {', '.join(subreddits)}")
        print(f"{'='*60}\n")
        
        # Track initial attempt count
        total_attempts = 0
        
        for query in queries:
            if len(data) >= candidate_target:
                break
                
            for subreddit in subreddits:
                if len(data) >= candidate_target:
                    break
                    
                for sort in sort_methods:
                    if len(data) >= candidate_target:
                        break
                        
                    for time_filter in time_filters:
                        if len(data) >= candidate_target:
                            break
                        
                        try:
                            results_this_query = 0
                            
                            # Use pagination to get all results
                            for post in self.reddit.subreddit(subreddit).search(
                                query, sort=sort, time_filter=time_filter, limit=250
                            ):
                                total_attempts += 1
                                
                                # Date filtering
                                post_date = datetime.fromtimestamp(post.created_utc)
                                if post_date < start_date or post_date > end_date:
                                    continue
                                
                                # Deduplication
                                if post.id in seen_ids:
                                    continue
                                
                                seen_ids.add(post.id)
                                results_this_query += 1
                                
                                # Extract comprehensive metadata
                                post_data = {
                                    # Core text data
                                    'id': post.id,
                                    'raw_text': f"{post.title} {post.selftext}".strip(),
                                    'title': post.title,
                                    'body': post.selftext,
                                    
                                    # Temporal data
                                    'created_utc': post_date,
                                    'created_date': post_date.strftime('%Y-%m-%d'),
                                    'created_year': post_date.year,
                                    'created_month': post_date.month,
                                    
                                    # Engagement metrics (public_metrics equivalent)
                                    'score': post.score,
                                    'upvote_ratio': post.upvote_ratio,
                                    'num_comments': post.num_comments,
                                    'engagement_total': post.score + post.num_comments,
                                    
                                    # Source metadata
                                    'subreddit': post.subreddit.display_name,
                                    'is_self': post.is_self,
                                    'is_video': post.is_video,
                                    'permalink': f"https://reddit.com{post.permalink}",
                                    
                                    # Anonymized author (IRB compliant)
                                    'author_id_hash': self._hash_user_id(
                                        str(post.author) if post.author else 'deleted'
                                    ),
                                    
                                    # Query tracking
                                    'search_query': query,
                                    'search_subreddit': subreddit
                                }
                                
                                # Optional: Get top comments
                                if include_comments:
                                    try:
                                        post.comments.replace_more(limit=0)
                                        top_comments = [c.body for c in post.comments[:3] 
                                                       if hasattr(c, 'body')]
                                        post_data['top_comments'] = " | ".join(top_comments)
                                    except:
                                        post_data['top_comments'] = ""
                                
                                data.append(post_data)
                                
                                if len(data) >= candidate_target:
                                    break
                            
                            # Log this query
                            self._log_query(query, subreddit, sort, 
                                           time_filter, results_this_query)
                            
                            if results_this_query > 0:
                                print(f"  [{len(data):>5}/{candidate_target}] "
                                      f"q='{query[:30]}...' r/{subreddit} "
                                      f"sort={sort} t={time_filter} → +{results_this_query}")
                                      
                        except Exception as e:
                            if "rate" in str(e).lower():
                                print(f"  ⏳ Rate limited, waiting 60s...")
                                time.sleep(60)
                            continue
        
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
        
        # Record attrition from extraction phase
        self.attrition_log['N0_raw_extracted'] = total_attempts
        self.attrition_log['N1_after_date_filter'] = len(df)
        self.attrition_log['N1_unique_posts'] = len(df)
        
        print(f"\n{'='*60}")
        print(f"✅ EXTRACTION COMPLETE")
        print(f"   Total API hits: {total_attempts}")
        print(f"   Unique posts collected: {len(df)}")
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
        
        # Generate extraction summary
        summary = {
            "extraction_date": timestamp,
            "total_posts": len(df),
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


def create_methane_dairy_queries() -> Tuple[List[str], List[str]]:
    """
    Create research queries for methane/dairy climate discourse.
    Following Boolean Logic Framework from methodology.
    """
    # Primary concepts (OR relationship)
    primary_concepts = [
        "methane emissions dairy",
        "methane cows climate",
        "cattle methane greenhouse",
        "dairy farming emissions",
        "livestock methane environment",
        "cow methane climate change",
        "enteric fermentation methane",
        "dairy industry carbon footprint",
        "methane reduction cattle",
        "factory farming methane",
        "regenerative dairy farming",
        "sustainable dairy farming",
        "methane digesters dairy",
        "cow burps climate",
        "beef cattle emissions"
    ]
    
    # Contextual constraints (AND relationship)
    contextual_constraints = [
        "climate",
        "environment", 
        "sustainability",
        "emissions",
        "greenhouse gas"
    ]
    
    return primary_concepts, contextual_constraints


def extract_reddit_data(target_count: int = 3000, 
                        start_year: int = 2018) -> pd.DataFrame:
    """
    Convenience function to run full extraction pipeline.
    
    Args:
        target_count: Target number of posts (2500-3000 recommended)
        start_year: Start year for data collection (2018 default)
    
    Returns:
        DataFrame with extracted posts
    """
    extractor = RedditDataExtractor()
    
    # Get predefined queries for methane/dairy research
    primary_concepts, contextual_constraints = create_methane_dairy_queries()
    
    # Construct query variations
    queries = extractor.construct_queries(
        primary_concepts=primary_concepts,
        contextual_constraints=contextual_constraints
    )
    
    print(f"📝 Generated {len(queries)} query variations")
    
    # Extract posts
    df = extractor.extract_posts(
        queries=queries,
        target_count=target_count,
        start_date=datetime(start_year, 1, 1),
        include_comments=False  # Set True if comment analysis needed
    )
    
    # Save data
    if len(df) > 0:
        extractor.save_data(df, output_dir="data", format="csv")
    
    return df, extractor


if __name__ == "__main__":
    # Run extraction
    df, extractor = extract_reddit_data(target_count=3000, start_year=2018)
    
    # Print summary
    if len(df) > 0:
        print("\n📊 EXTRACTION SUMMARY:")
        print(f"   Total posts: {len(df)}")
        print(f"   Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        print(f"   Unique subreddits: {df['subreddit'].nunique()}")
        print(f"\n   Top subreddits:")
        print(df['subreddit'].value_counts().head(10).to_string())
        print(f"\n   Posts per year:")
        print(df['created_year'].value_counts().sort_index().to_string())
