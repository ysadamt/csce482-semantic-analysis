import praw
import pandas as pd
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

# --- NLTK Setup ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

# ==========================================
# PART 1: CONNECT TO REAL API (REDDIT)
# ==========================================
def scrape_reddit_data():
    print("--- Connecting to Reddit API ---")
    
    # ⚠️ PASTE YOUR CREDENTIALS HERE ⚠️
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_KEY"),      # e.g., 'AbCdEfG123'
        client_secret=os.getenv("REDDIT_SECRET"), # e.g., '12345-AbCd...'
        user_agent="methane_scraper_v1"       # Name of your bot
    )

    # We will search for specific keywords across all of Reddit
    # or specific subreddits like r/farming or r/environment
    keywords = 'methane dairy cows environment climate'
    limit = 500
    
    print(f"Searching for '{keywords}' (Limit: {limit} posts)...")
    
    posts_data = []
    
    try:
        # Searching all subreddits ('all') for the keywords
        for post in reddit.subreddit('all').search(keywords, limit=limit):
            
            # Combine title and body text for better sentiment context
            full_text = f"{post.title} {post.selftext}"
            
            posts_data.append({
                "source": f"r/{post.subreddit.display_name}",
                "text": full_text,
                "score": post.score # How many upvotes it has
            })
            
    except Exception as e:
        print(f"ERROR: Could not connect to Reddit. Check your API Keys. Details: {e}")
        return pd.DataFrame() # Return empty if fails

    df = pd.DataFrame(posts_data)
    print(f"Successfully scraped {len(df)} real posts.")
    return df

# ==========================================
# PART 2: THE 9-STEP CLEANING PIPELINE
# ==========================================
def process_data(df):
    if df.empty:
        print("No data to process.")
        return df, Counter()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Domain specific noise words
    custom_stops = {'stop', 'now', 'click', 'removed', 'deleted'} 
    stop_words.update(custom_stops)

    print("\n--- Running 9-Step Cleaning Pipeline ---")

    # STEP 1: Remove duplicate posts
    df.drop_duplicates(subset='text', inplace=True)

    # STEP 2: Remove usernames and links
    df['clean_text'] = df['text'].apply(lambda x: re.sub(r'(@\w+|http\S+)', '', str(x)))

    # STEP 3: Remove special characters (keeping # for Step 6)
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s#]', '', x))

    # STEP 4: Exclude meaningless words
    df['clean_text'] = df['clean_text'].apply(
        lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
    )

    # STEP 5: Save text for sentiment analysis (Checkpoint)
    df['sentiment_text'] = df['clean_text']

    # STEP 6: Remove hashtagged words
    df['final_tokens'] = df['clean_text'].apply(lambda x: re.sub(r'#\w+', '', x))

    # STEP 7: Tokenize
    df['tokens'] = df['final_tokens'].apply(lambda x: nltk.word_tokenize(x.lower()))

    # STEP 9: Convert to base form (Lemmatization)
    df['lemmatized'] = df['tokens'].apply(
        lambda tokens: [lemmatizer.lemmatize(word) for word in tokens if word.strip()]
    )

    # STEP 8: Count Bigrams
    all_words = [word for tokens in df['lemmatized'] for word in tokens]
    bigrams = list(ngrams(all_words, 2))
    bigram_counts = Counter(bigrams)
    
    return df, bigram_counts

# ==========================================
# PART 3: SENTIMENT ANALYSIS
# ==========================================
def analyze_sentiment(df, bigram_counts):
    if df.empty: return

    print("\n--- Sentiment Analysis Results ---")
    
    # Calculate Polarity
    df['polarity'] = df['sentiment_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Labeling
    def get_label(score):
        if score < -0.05: return 'Negative' # Slight buffer for "Neutral"
        if score > 0.05: return 'Positive'
        return 'Neutral'

    df['sentiment_label'] = df['polarity'].apply(get_label)

    # Visualization of Text Data
    print(f"\nTotal Posts Analyzed: {len(df)}")
    print("\nTOP 5 MOST COMMON PHRASES (Bigrams):")
    # This reveals what people are actually talking about
    for bigram, count in bigram_counts.most_common(5):
        print(f"{bigram[0]} {bigram[1]}: {count}")

    print("\nSENTIMENT DISTRIBUTION:")
    print(df['sentiment_label'].value_counts())

    print("\nSAMPLE DATA:")
    print(df[['source', 'sentiment_label', 'text']].head(10))

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Scrape Real Data
    df = scrape_reddit_data()
    
    # 2. Clean
    df_clean, bigrams = process_data(df)
    
    # 3. Analyze
    analyze_sentiment(df_clean, bigrams)