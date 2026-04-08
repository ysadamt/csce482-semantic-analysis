"""
Full Research Pipeline Runner
Methane/Dairy Climate Discourse Analysis from Reddit

This script executes the complete 3-phase research pipeline:
- Phase 1: Data Extraction (Reddit API)
- Phase 2: Data Preprocessing (cleaning, normalization)
- Phase 3: Empirical Analysis (trends, spikes, odds ratios)

Usage:
    python run_analysis.py --extract          # Run data extraction only
    python run_analysis.py --preprocess       # Run preprocessing only
    python run_analysis.py --analyze          # Run analysis only
    python run_analysis.py --full             # Run complete pipeline
    python run_analysis.py --from-csv FILE    # Load existing CSV and analyze
"""

import argparse
import os
import pandas as pd
from datetime import datetime
import json
import sys

# Import our modules
from data_extractor import RedditDataExtractor, create_methane_dairy_queries, extract_reddit_data
from preprocessor import DataPreprocessor, preprocess_reddit_data
from analysis import run_full_analysis, SentimentAnalyzer
from statistical_analysis import run_statistical_analysis


def setup_directories():
    """Create output directories if they don't exist."""
    dirs = ['data', 'results', 'figures', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def phase1_extraction(target_count: int = 3000, 
                     start_year: int = 2018,
                     include_comments: bool = True,
                     max_comments_per_post: int = 10,
                     fast_mode: bool = False,
                     save: bool = True) -> pd.DataFrame:
    """
    Execute Phase 1: Reddit Data Extraction.
    
    Args:
        target_count: Target output rows (posts + comments, recommended: 2500-3000)
        start_year: Start year for data collection
        include_comments: Whether to include comment text in extraction output
        max_comments_per_post: Maximum comments to capture per post
        fast_mode: Reduce search breadth for quicker extraction
        save: Whether to save the extracted data
        
    Returns:
        DataFrame with extracted rows
    """
    print("\n" + "="*70)
    print("PHASE 1: DATA EXTRACTION (REDDIT-ONLY)")
    print("="*70)
    
    try:
        df, extractor = extract_reddit_data(
            target_count=target_count,
            start_year=start_year,
            include_comments=include_comments,
            max_comments_per_post=max_comments_per_post,
            fast_mode=fast_mode,
        )
        
        if len(df) == 0:
            print("⚠️ No data extracted. Check your Reddit API credentials.")
            print("   Set REDDIT_KEY and REDDIT_SECRET in your .env file")
            return None
        
        # Print extraction summary
        total_rows = len(df)
        if 'content_type' in df.columns:
            content_types = df['content_type'].fillna('post').astype(str).str.lower()
            post_rows = int((content_types == 'post').sum())
            comment_rows = int((content_types == 'comment').sum())
            post_mask = content_types == 'post'
        else:
            post_rows = total_rows
            comment_rows = 0
            post_mask = pd.Series(True, index=df.index)

        print("\n📊 EXTRACTION SUMMARY:")
        print(f"   Total rows: {total_rows}")
        print(f"   Post rows: {post_rows}")
        print(f"   Comment rows: {comment_rows}")
        print(f"   Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        print(f"   Unique subreddits: {df['subreddit'].nunique()}")
        if 'comments_collected_count' in df.columns and post_rows > 0:
            print(
                f"   Avg comments captured/post: "
                f"{df.loc[post_mask, 'comments_collected_count'].mean():.2f}"
            )
        
        # Print attrition report
        print("\n📋 ATTRITION REPORT:")
        print(extractor.get_attrition_report().to_string(index=False))
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/reddit_extracted_{timestamp}.csv"
            df.to_csv(output_path, index=False)
            print(f"\n💾 Data saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        print("\nTroubleshooting:")
        print("1. Create a .env file with REDDIT_KEY and REDDIT_SECRET")
        print("2. Get API credentials at: https://www.reddit.com/prefs/apps")
        return None


def phase2_preprocessing(df: pd.DataFrame, 
                        include_comments_in_text: bool = True,
                        save: bool = True) -> pd.DataFrame:
    """
    Execute Phase 2: Data Preprocessing.
    
    Args:
        df: DataFrame from Phase 1
        include_comments_in_text: Merge comments into analysis text if available
        save: Whether to save preprocessed data
        
    Returns:
        Preprocessed DataFrame
    """
    print("\n" + "="*70)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*70)
    
    if df is None or len(df) == 0:
        print("⚠️ No data to preprocess")
        return None
    
    df_clean, preprocessor = preprocess_reddit_data(
        df,
        include_comments_in_text=include_comments_in_text
    )
    
    # Print final statistics
    print("\n📊 PREPROCESSING RESULTS:")
    print(f"   Rows retained: {len(df_clean)} / {len(df)} "
          f"({len(df_clean)/len(df)*100:.1f}%)")
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/reddit_preprocessed_{timestamp}.csv"
        df_clean.to_csv(output_path, index=False)
        
        # Save attrition table
        attrition_path = f"data/attrition_report_{timestamp}.csv"
        preprocessor.get_attrition_table().to_csv(attrition_path, index=False)
        
        print(f"\n💾 Preprocessed data saved to: {output_path}")
        print(f"💾 Attrition report saved to: {attrition_path}")
    
    return df_clean


def phase1b_comment_enrichment(input_csv: str = None,
                               max_comments_per_post: int = 10,
                               skip_existing_comments: bool = True,
                               save: bool = True) -> pd.DataFrame:
    """
    Enrich an existing extracted post dataset with comments by post ID.

    This does not rescrape posts; it only fetches comments for rows already present.

    Args:
        input_csv: Existing extraction CSV path; defaults to latest reddit_extracted*.csv
        max_comments_per_post: Maximum comments to capture per post
        skip_existing_comments: Skip rows that already have comments_text
        save: Whether to save enriched CSV and enrichment log

    Returns:
        DataFrame enriched with comments_text and comments_collected_count
    """
    print("\n" + "="*70)
    print("PHASE 1B: COMMENT ENRICHMENT (NO POST RESCRAPE)")
    print("="*70)

    if input_csv:
        source_path = input_csv
    else:
        data_files = [f for f in os.listdir('data') if f.startswith('reddit_extracted')]
        if not data_files:
            print("❌ No extraction file found. Run --extract first or specify --input")
            return None
        source_path = os.path.join('data', sorted(data_files)[-1])
        print(f"📂 Using latest extraction: {os.path.basename(source_path)}")

    try:
        df = pd.read_csv(source_path)
    except Exception as e:
        print(f"❌ Failed to read input CSV: {e}")
        return None

    if len(df) == 0:
        print("⚠️ Input CSV is empty; nothing to enrich")
        return df

    extractor = RedditDataExtractor()
    df_enriched = extractor.enrich_comments_for_dataframe(
        df,
        max_comments_per_post=max_comments_per_post,
        skip_existing_comments=skip_existing_comments,
    )

    print("\n📊 COMMENT ENRICHMENT SUMMARY:")
    print(f"   Rows: {len(df_enriched)}")
    if 'content_type' in df_enriched.columns:
        content_types = df_enriched['content_type'].fillna('post').astype(str).str.lower()
        post_rows = int((content_types == 'post').sum())
        comment_rows = int((content_types == 'comment').sum())
        post_mask = content_types == 'post'
        print(f"   Post rows: {post_rows}")
        print(f"   Comment rows: {comment_rows}")
    else:
        post_rows = len(df_enriched)
        post_mask = pd.Series(True, index=df_enriched.index)

    if 'comments_collected_count' in df_enriched.columns and post_rows > 0:
        print(
            f"   Avg comments captured/post: "
            f"{df_enriched.loc[post_mask, 'comments_collected_count'].mean():.2f}"
        )

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/reddit_comments_enriched_{timestamp}.csv"
        df_enriched.to_csv(output_path, index=False)

        enrich_log_path = f"data/reddit_comment_enrichment_{timestamp}.json"
        with open(enrich_log_path, 'w') as f:
            json.dump(extractor.attrition_log, f, indent=2)

        print(f"\n💾 Comment-enriched data saved to: {output_path}")
        print(f"💾 Enrichment log saved to: {enrich_log_path}")

    return df_enriched


def phase3_analysis(df: pd.DataFrame, 
                   output_dir: str = 'results',
                   policy_date: str = '2024-01-01') -> dict:
    """
    Execute Phase 3: Empirical Analysis.
    
    Args:
        df: Preprocessed DataFrame
        output_dir: Directory for outputs
        policy_date: Date string (YYYY-MM-DD) to split Before/After groups
        
    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*70)
    print("PHASE 3: EMPIRICAL ANALYSIS")
    print("="*70)
    
    if df is None or len(df) == 0:
        print("⚠️ No data to analyze")
        return None
    
    # Ensure required columns exist
    if 'clean_text' not in df.columns and 'raw_text' in df.columns:
        df['clean_text'] = df['raw_text']
    
    results, df_analyzed = run_full_analysis(
        df,
        text_column='clean_text',
        date_column='created_utc',
        output_dir=output_dir
    )
    
    # Save final analyzed DataFrame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_analyzed.to_csv(f"{output_dir}/analyzed_data_{timestamp}.csv", index=False)
    
    return results, df_analyzed


def phase4_statistical_analysis(df: pd.DataFrame,
                                output_dir: str = 'results',
                                policy_date: str = '2024-01-01') -> dict:
    """
    Execute Phase 4: Publication-Ready Statistical Analysis.
    
    Covers every checklist item:
    - Data quality & relevance
    - Hypothesis testing (H₀/H₁, normality, t-test or Mann-Whitney, Cohen's d, CI)
    - Bias assessment (geographic, platform, time-window, user concentration)
    - Sensitivity analysis (excluding top 5 % users)
    - Drivers of sentiment (logistic regression odds ratios)
    
    Args:
        df: Analyzed DataFrame (must contain sentiment_score & sentiment columns)
        output_dir: Directory for outputs
        policy_date: Date string for Before/After split
        
    Returns:
        Dictionary with statistical analysis results
    """
    print("\n" + "="*70)
    print("PHASE 4: STATISTICAL ANALYSIS (PUBLICATION CHECKLIST)")
    print("="*70)
    
    if df is None or len(df) == 0:
        print("⚠️ No data to analyze")
        return None
    
    stat_results = run_statistical_analysis(
        df,
        text_column='clean_text',
        date_column='created_utc',
        policy_date=policy_date,
        output_dir=output_dir,
    )
    
    return stat_results


def run_full_pipeline(target_count: int = 3000,
                     start_year: int = 2018,
                     include_comments: bool = True,
                     max_comments_per_post: int = 10,
                     fast_mode: bool = False,
                     include_comments_in_text: bool = True) -> dict:
    """
    Run the complete research pipeline (Phases 1-3).
    
    Args:
        target_count: Target output rows (posts + comments)
        start_year: Start year for data collection
        include_comments: Include comments in extraction output
        max_comments_per_post: Maximum comments captured per post
        fast_mode: Reduce extraction search breadth for faster runtime
        include_comments_in_text: Merge comments into analysis text in preprocessing
        
    Returns:
        Dictionary with all results
    """
    print("\n" + "="*70)
    print("FULL RESEARCH PIPELINE")
    print("Methane/Dairy Climate Discourse Analysis")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target rows (posts + comments): {target_count}")
    print(f"Date range: {start_year}-01-01 to present")
    
    setup_directories()
    
    # Phase 1: Extraction
    df_raw = phase1_extraction(
        target_count=target_count,
        start_year=start_year,
        include_comments=include_comments,
        max_comments_per_post=max_comments_per_post,
        fast_mode=fast_mode,
    )
    
    if df_raw is None:
        print("\n❌ Pipeline stopped: Extraction failed")
        return None
    
    # Phase 2: Preprocessing
    df_clean = phase2_preprocessing(
        df_raw,
        include_comments_in_text=include_comments_in_text
    )
    
    if df_clean is None:
        print("\n❌ Pipeline stopped: Preprocessing failed")
        return None
    
    # Phase 3: Analysis
    phase3_out = phase3_analysis(df_clean)
    if phase3_out is None:
        print("\n❌ Pipeline stopped: Analysis failed")
        return None
    results, df_analyzed = phase3_out
    
    # Phase 4: Statistical Analysis (publication checklist)
    stat_results = phase4_statistical_analysis(
        df_analyzed, output_dir='results', policy_date='2024-01-01'
    )
    if stat_results is not None:
        results['statistical_analysis'] = stat_results
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs saved in:")
    print(f"   📁 data/      - Raw and preprocessed data")
    print(f"   📁 results/   - Analysis results and figures")
    
    return results


def load_and_analyze(csv_path: str) -> dict:
    """
    Load existing CSV and run analysis pipeline.
    
    Args:
        csv_path: Path to CSV file with Reddit data
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n📂 Loading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Try to identify the text column
    text_cols = ['raw_text', 'clean_text', 'text', 'body', 'content']
    text_col = None
    for col in text_cols:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print(f"❌ Could not find text column. Available columns: {df.columns.tolist()}")
        return None
    
    # Try to identify date column
    date_cols = ['created_utc', 'created_at', 'date', 'timestamp']
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            df[date_col] = pd.to_datetime(df[date_col])
            break
    
    print(f"   Found text column: {text_col}")
    print(f"   Found date column: {date_col}")
    print(f"   Total rows: {len(df)}")
    
    setup_directories()
    
    # Check if preprocessing is needed
    if 'clean_text' not in df.columns:
        print("\n⚡ Running preprocessing...")
        df, _ = preprocess_reddit_data(df)
    
    # Run analysis
    if date_col:
        df['created_utc'] = df[date_col]
    else:
        # Create fake dates for analysis
        df['created_utc'] = pd.date_range('2020-01-01', periods=len(df), freq='H')
    
    phase3_out = phase3_analysis(df)
    if phase3_out is None:
        return None
    results, df_analyzed = phase3_out
    
    # Phase 4: Statistical Analysis
    stat_results = phase4_statistical_analysis(df_analyzed)
    if stat_results is not None:
        results['statistical_analysis'] = stat_results
    
    return results


def print_usage_guide():
    """Print usage guide for the pipeline."""
    guide = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║        REDDIT SENTIMENT ANALYSIS RESEARCH PIPELINE               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  SETUP:                                                          ║
    ║  1. Create a .env file with your Reddit API credentials:         ║
    ║     REDDIT_KEY=your_client_id                                    ║
    ║     REDDIT_SECRET=your_client_secret                             ║
    ║                                                                  ║
    ║  2. Get credentials at: https://www.reddit.com/prefs/apps        ║
    ║     - Create a "script" type application                         ║
    ║                                                                  ║
    ║  USAGE:                                                          ║
    ║  python run_analysis.py --full           Run complete pipeline   ║
    ║  python run_analysis.py --extract        Extract data only       ║
    ║  python run_analysis.py --enrich-comments Add comments only       ║
    ║  python run_analysis.py --preprocess     Preprocess only         ║
    ║  python run_analysis.py --analyze        Analyze only            ║
    ║  python run_analysis.py --from-csv FILE  Analyze existing CSV    ║
    ║                                                                  ║
    ║  OPTIONS:                                                        ║
    ║  --target N      Target rows (posts + comments, default: 3000)   ║
    ║  --start-year Y  Start year for data (default: 2018)             ║
    ║  --fast-mode     Faster extraction with reduced search breadth    ║
    ║  --disable-comments Disable Reddit comment scraping               ║
    ║  --max-comments-per-post N  Max comments/post (default: 10)      ║
    ║  --refresh-existing-comments  Refetch even if comments exist      ║
    ║  --analyze-after-enrich  Run preprocess+analysis after enrich     ║
    ║  --exclude-comments-from-analysis  Use post text only             ║
    ║                                                                  ║
    ║  OUTPUTS:                                                        ║
    ║  📁 data/      - Raw and preprocessed data files                 ║
    ║  📁 results/   - Analysis results, figures, and statistics       ║
    ║                                                                  ║
    ║  ANALYSIS TECHNIQUES:                                            ║
    ║  1. Trend-Based Temporal Analysis                                ║
    ║  2. Statistical Spike Detection                                  ║
    ║  3. Odds Ratio / Log-Odds Semantic Analysis                      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(guide)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Reddit Sentiment Analysis Research Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_analysis.py --full                    # Run complete pipeline
    python run_analysis.py --full --target 2500     # Extract 2500 rows
    python run_analysis.py --extract --fast-mode    # Faster extraction mode
    python run_analysis.py --enrich-comments --input data/reddit_extracted_x.csv
                                                                                                     # Add comments to existing rows
    python run_analysis.py --enrich-comments --analyze-after-enrich
                                                                                                     # Enrich comments then analyze
  python run_analysis.py --from-csv data.csv      # Analyze existing data
        '''
    )
    
    # Pipeline stage options
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument('--full', action='store_true',
                            help='Run complete pipeline (extract → preprocess → analyze)')
    stage_group.add_argument('--extract', action='store_true',
                            help='Run data extraction only')
    stage_group.add_argument('--enrich-comments', action='store_true',
                            help='Fetch comments for existing post rows only (no rescrape)')
    stage_group.add_argument('--preprocess', action='store_true',
                            help='Run preprocessing only (requires existing data)')
    stage_group.add_argument('--analyze', action='store_true',
                            help='Run analysis only (requires preprocessed data)')
    stage_group.add_argument('--from-csv', type=str, metavar='FILE',
                            help='Load existing CSV and run analysis')
    stage_group.add_argument('--help-usage', action='store_true',
                            help='Show detailed usage guide')
    
    # Configuration options
    parser.add_argument('--target', type=int, default=3000,
                       help='Target output rows (posts + comments, default: 3000)')
    parser.add_argument('--start-year', type=int, default=2018,
                       help='Start year for data extraction (default: 2018)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Enable faster extraction with reduced search breadth')
    parser.add_argument('--disable-comments', action='store_true',
                       help='Disable comment scraping during extraction (default: enabled)')
    parser.add_argument('--max-comments-per-post', type=int, default=10,
                       help='Maximum comments captured per post (default: 10)')
    parser.add_argument('--refresh-existing-comments', action='store_true',
                       help='Refetch comments even when comments_text already exists')
    parser.add_argument('--analyze-after-enrich', action='store_true',
                       help='After --enrich-comments, run preprocess and analysis')
    parser.add_argument('--exclude-comments-from-analysis', action='store_true',
                       help='Do not merge comments into analysis text during preprocessing')
    parser.add_argument('--policy-date', type=str, default='2024-01-01',
                       help='Policy date for Before/After comparison (default: 2024-01-01)')
    parser.add_argument('--input', type=str,
                       help='Input CSV file for preprocessing or analysis')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    # Show usage guide
    if args.help_usage or len(sys.argv) == 1:
        print_usage_guide()
        if len(sys.argv) == 1:
            print("\nRun with --help for command options, or --full to start the pipeline.")
        return
    
    # Execute requested stage
    if args.full:
        run_full_pipeline(
            target_count=args.target,
            start_year=args.start_year,
            include_comments=not args.disable_comments,
            max_comments_per_post=args.max_comments_per_post,
            fast_mode=args.fast_mode,
            include_comments_in_text=not args.exclude_comments_from_analysis
        )
        
    elif args.extract:
        phase1_extraction(
            target_count=args.target,
            start_year=args.start_year,
            include_comments=not args.disable_comments,
            max_comments_per_post=args.max_comments_per_post,
            fast_mode=args.fast_mode,
        )

    elif args.enrich_comments:
        df_enriched = phase1b_comment_enrichment(
            input_csv=args.input,
            max_comments_per_post=args.max_comments_per_post,
            skip_existing_comments=not args.refresh_existing_comments,
        )

        if df_enriched is not None and args.analyze_after_enrich:
            df_clean = phase2_preprocessing(
                df_enriched,
                include_comments_in_text=not args.exclude_comments_from_analysis
            )
            if df_clean is not None:
                phase3_out = phase3_analysis(
                    df_clean,
                    output_dir=args.output_dir,
                    policy_date=args.policy_date
                )
                if phase3_out is not None:
                    _, df_analyzed = phase3_out
                    phase4_statistical_analysis(
                        df_analyzed,
                        output_dir=args.output_dir,
                        policy_date=args.policy_date
                    )
        
    elif args.preprocess:
        if args.input:
            df = pd.read_csv(args.input)
            phase2_preprocessing(
                df,
                include_comments_in_text=not args.exclude_comments_from_analysis
            )
        else:
            # Find most recent extraction file
            data_files = [f for f in os.listdir('data') if f.startswith('reddit_extracted')]
            if data_files:
                latest = sorted(data_files)[-1]
                df = pd.read_csv(f'data/{latest}')
                print(f"📂 Using latest extraction: {latest}")
                phase2_preprocessing(
                    df,
                    include_comments_in_text=not args.exclude_comments_from_analysis
                )
            else:
                print("❌ No extraction file found. Run --extract first or specify --input")
                
    elif args.analyze:
        if args.input:
            df = pd.read_csv(args.input)
        else:
            # Find most recent preprocessed file
            data_files = [f for f in os.listdir('data') if f.startswith('reddit_preprocessed')]
            if data_files:
                latest = sorted(data_files)[-1]
                df = pd.read_csv(f'data/{latest}')
                print(f"📂 Using latest preprocessed data: {latest}")
            else:
                print("❌ No preprocessed file found. Run --preprocess first or specify --input")
                return
        phase3_out = phase3_analysis(df, output_dir=args.output_dir,
                                     policy_date=args.policy_date)
        if phase3_out is not None:
            _, df_analyzed = phase3_out
            phase4_statistical_analysis(df_analyzed, output_dir=args.output_dir,
                                        policy_date=args.policy_date)
                
    elif args.from_csv:
        load_and_analyze(args.from_csv)


if __name__ == "__main__":
    main()
