import praw
import pandas as pd
import re
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from dotenv import load_dotenv
import os

load_dotenv()

# --- Setup NLTK ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# ======================================================
# PART 1: REDDIT SCRAPER (Target: Methane & Dairy)
# ======================================================
def step_1_scrape_reddit():
    print("\n--- STEP 1: SCRAPING REDDIT ---")
    
    # ⚠️ REPLACE WITH YOUR ACTUAL CREDENTIALS ⚠️
    # If you don't have them yet, the code will fail gracefully.
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_KEY"),
            client_secret=os.getenv("REDDIT_SECRET"),
            user_agent="methane_research_bot_v1"
        )
        
        # Multiple query variations to bypass the ~250 result-per-query cap
        queries = [
            'methane cows climate emission dairy industry',
            'methane cattle greenhouse gas',
            'dairy farming emissions climate change',
            'cow methane enteric fermentation',
            'livestock emissions global warming',
            'methane reduction dairy farm',
            'factory farming methane environment',
            'cattle ranching climate impact',
        ]
        subreddits = ['all', 'climate', 'environment', 'farming', 'science']
        sort_methods = ['relevance', 'top', 'comments']
        time_filters = ['all', 'year', 'month']
        
        target = 3000
        seen_ids = set()
        data = []
        cutoff_date = datetime(2016, 1, 1)
        
        print(f"Searching Reddit with multiple queries to reach {target} posts (2016 onwards)...")
        
        for query in queries:
            if len(data) >= target:
                break
            for subreddit in subreddits:
                if len(data) >= target:
                    break
                for sort in sort_methods:
                    if len(data) >= target:
                        break
                    for time_filter in time_filters:
                        if len(data) >= target:
                            break
                        try:
                            for post in reddit.subreddit(subreddit).search(
                                query, sort=sort, time_filter=time_filter, limit=250
                            ):
                                post_date = datetime.fromtimestamp(post.created_utc)
                                if post.id not in seen_ids and post_date >= cutoff_date:
                                    seen_ids.add(post.id)
                                    data.append({
                                        'text': f"{post.title} {post.selftext}",
                                        'score': post.score,
                                        'id': post.id,
                                        'created_utc': datetime.fromtimestamp(post.created_utc)
                                    })
                            print(f"  [{len(data):>5} posts] q='{query[:40]}...' r/{subreddit} sort={sort} t={time_filter}")
                        except Exception:
                            pass  # skip failed queries silently
        
        df = pd.DataFrame(data)
        print(f"✅ Success! Scraped {len(df)} unique posts.")
        return df

    except Exception as e:
        print(f"⚠️ API Error (using mock data for demonstration): {e}")
        # MOCK DATA if API fails (so you can test the rest of the code)
        mock_data = [
            "Methane from dairy cows is a major greenhouse gas problem.",
            "New anaerobic digesters are reducing emissions on farms.",
            "I think the methane issue is exaggerated by activists.",
            "Cows produce methane through enteric fermentation.",
            "We need to stop factory farming to save the climate.",
            "Digestors turn methane into renewable energy which is good.",
            "Climate change is a hoax, leave the farmers alone.",
            "Methane is 80x more potent than CO2 in the short term."
        ] * 125
        # Generate mock dates spanning 24 months for temporal analysis
        base_date = datetime(2024, 1, 1)
        mock_dates = [base_date + timedelta(days=int(i * 730 / 1000)) for i in range(1000)]
        return pd.DataFrame({'text': mock_data, 'score': [10]*1000, 'created_utc': mock_dates})

# ======================================================
# PART 2: LOAD PRETRAINED SENTIMENT MODEL (RoBERTa)
# ======================================================
def preprocess_tweet(text):
    """Preprocess text for twitter-roberta: replace @mentions and URLs."""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def step_2_load_model():
    print("\n--- STEP 2: LOADING PRETRAINED SENTIMENT MODEL ---")
    
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"Loading model: {MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()  # Set to evaluation mode
    
    # Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
    print(f"Labels: {config.id2label}")
    print("✅ Pretrained RoBERTa model loaded.")
    
    return model, tokenizer, config

# ======================================================
# MODIFIED PART 3: NETWORK ANALYSIS (IMPROVED VISUALS)
# ======================================================
def step_3_network_analysis(df):
    print("\n--- STEP 3: NETWORK ANALYSIS (KEYWORD MAPPING) ---")
    
    if df.empty:
        print("⚠️ No data available for network analysis.")
        return

    # 1. Pre-clean for network mapping
    stop_words = set(stopwords.words('english'))
    additional_stops = {
        'methane', 'dairy', 'http', 'https', 'cows', 'industry', 
        'would', 'could', 'should', 'people', 'like', 'make', 'think'
    }
    stop_words.update(additional_stops)
    lemmatizer = WordNetLemmatizer()
    
    # 2. Extract co-occurrences and word frequencies
    co_occurrence = Counter()
    word_freq = Counter()
    
    for text in df['text']:
        clean = re.sub(r'[^\w\s]', '', text.lower())
        tokens = [lemmatizer.lemmatize(w) for w in clean.split() if w not in stop_words and len(w) > 2]
        
        for token in tokens:
            word_freq[token] += 1
        
        unique_tokens = sorted(list(set(tokens)))
        for i in range(len(unique_tokens)):
            for j in range(i + 1, len(unique_tokens)):
                edge = (unique_tokens[i], unique_tokens[j])
                co_occurrence[edge] += 1

    # 3. Build Graph with NetworkX
    G = nx.Graph()
    
    num_edges_added = 0
    for (w1, w2), weight in co_occurrence.most_common(200):
        if weight > 1: 
            G.add_edge(w1, w2, weight=weight)
            num_edges_added += 1
        if num_edges_added >= 100:
            break
            
    print(f"Network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if G.number_of_nodes() == 0:
        print("⚠️ Not enough significant co-occurrences to plot.")
        return
    
    # 4. Community Detection (Louvain method)
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G, weight='weight'))
    
    # Assign community ID to each node
    node_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_community[node] = idx
    
    num_communities = len(communities)
    print(f"Detected {num_communities} topic communities.")
    
    # Print community details
    for idx, community in enumerate(communities):
        # Sort words in community by frequency
        sorted_words = sorted(community, key=lambda w: word_freq.get(w, 0), reverse=True)
        top_words = sorted_words[:5]
        print(f"  Community {idx + 1}: {', '.join(top_words)} (+{max(0, len(community) - 5)} more)")
    
    # Print top 15 most frequent keywords in the network
    network_words = set(G.nodes())
    top_keywords = [(w, word_freq[w]) for w in network_words]
    top_keywords.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 15 Keywords by Frequency:")
    for word, freq in top_keywords[:15]:
        print(f"  {word}: {freq} occurrences")
    
    # ===================================================
    # VISUALIZATION WITH COMMUNITIES
    # ===================================================
    # Use distinct colors for communities
    community_palette = [
        '#4d96ff', '#ff6b6b', '#6bcb77', '#ffd93d', '#c084fc',
        '#f97316', '#06b6d4', '#ec4899', '#84cc16', '#a78bfa',
        '#fb923c', '#22d3ee', '#f472b6', '#a3e635', '#818cf8'
    ]
    
    plt.figure(figsize=(14, 11), facecolor='white')
    
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Node sizing based on word frequency
    max_freq = max(word_freq.get(n, 1) for n in G.nodes())
    node_sizes = [300 + (word_freq.get(n, 1) / max_freq) * 2000 for n in G.nodes()]
    
    # Node coloring based on community
    node_colors = [community_palette[node_community[n] % len(community_palette)] for n in G.nodes()]
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=1.0,
        edge_color='#cccccc',
        alpha=0.4
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        edgecolors='#333333',
        linewidths=1.0
    )
    
    # Labels: larger font for high-frequency words
    labels_large = {n: n for n in G.nodes() if word_freq.get(n, 0) >= top_keywords[min(9, len(top_keywords)-1)][1]}
    labels_small = {n: n for n in G.nodes() if n not in labels_large}
    
    nx.draw_networkx_labels(
        G, pos, labels=labels_large,
        font_size=12, font_family='sans-serif',
        font_weight='bold', font_color='#111111'
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels_small,
        font_size=8, font_family='sans-serif',
        font_weight='normal', font_color='#444444'
    )
    
    # Add legend for communities
    legend_handles = []
    for idx, community in enumerate(communities):
        sorted_words = sorted(community, key=lambda w: word_freq.get(w, 0), reverse=True)
        label = f"Topic {idx+1}: {', '.join(sorted_words[:3])}"
        color = community_palette[idx % len(community_palette)]
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=label))
    
    plt.legend(handles=legend_handles, loc='upper left', fontsize=9,
               framealpha=0.9, title='Topic Communities', title_fontsize=10)
    
    plt.title("Keyword Co-occurrence Network with Topic Communities",
              fontsize=14, fontweight='bold', color='#222222')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("✅ Network Map Generated.")

# ======================================================
# PART 4: FINAL SENTIMENT CLASSIFICATION
# ======================================================
def step_4_final_analysis(df, model, tokenizer, config):
    print("\n--- STEP 4: FINAL CLASSIFICATION ---")
    
    # Classify in batches to avoid memory issues
    batch_size = 32
    all_labels = []
    all_scores = []
    texts = df['text'].tolist()
    
    print(f"Classifying {len(texts)} posts in batches of {batch_size}...")
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [preprocess_tweet(t)[:512] for t in texts[i:i+batch_size]]
            encoded = tokenizer(batch_texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=512)
            output = model(**encoded)
            scores_batch = output.logits.detach().numpy()
            
            for scores_row in scores_batch:
                probs = softmax(scores_row)
                pred_idx = np.argmax(probs)
                all_labels.append(config.id2label[pred_idx])
                all_scores.append(float(np.max(probs)))
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} posts...")
    
    # Map labels to readable names
    label_display = {'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}
    df['predicted_sentiment'] = [label_display.get(l.lower(), l) for l in all_labels]
    df['confidence'] = all_scores
    
    # 3. Results
    print("\nFINAL RESULTS SUMMARY:")
    sentiment_counts = df['predicted_sentiment'].value_counts()
    print(sentiment_counts)
    
    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    
    print("\nSAMPLE CLASSIFICATIONS:")
    print(df[['text', 'predicted_sentiment', 'confidence']].head(5).to_string())
    
    # --- Visualization: Sentiment Analysis Results ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Define colors for each sentiment
    color_map = {'Negative': '#ff6b6b', 'Neutral': '#ffd93d', 'Positive': '#6bcb77'}
    colors = [color_map.get(label, '#999999') for label in sentiment_counts.index]
    
    # 1. Pie Chart - Sentiment Distribution
    axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, explode=[0.02]*len(sentiment_counts), shadow=True,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0].set_title('Sentiment Distribution (Pie)', fontsize=14, fontweight='bold')
    
    # 2. Bar Chart - Sentiment Counts
    bars = axes[1].bar(sentiment_counts.index, sentiment_counts.values, color=colors, edgecolor='#333333')
    axes[1].set_xlabel('Sentiment', fontsize=12)
    axes[1].set_ylabel('Number of Posts', fontsize=12)
    axes[1].set_title('Sentiment Distribution (Bar)', fontsize=14, fontweight='bold')
    # Add value labels on bars
    for bar, val in zip(bars, sentiment_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     str(val), ha='center', fontweight='bold', fontsize=11)
    
    # 3. Horizontal Bar Chart with Percentage
    total = sentiment_counts.sum()
    percentages = (sentiment_counts.values / total) * 100
    y_pos = range(len(sentiment_counts))
    bars_h = axes[2].barh(y_pos, percentages, color=colors, edgecolor='#333333')
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(sentiment_counts.index, fontsize=11)
    axes[2].set_xlabel('Percentage (%)', fontsize=12)
    axes[2].set_title('Sentiment Percentage', fontsize=14, fontweight='bold')
    # Add percentage labels
    for bar, pct in zip(bars_h, percentages):
        axes[2].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{pct:.1f}%', va='center', fontweight='bold', fontsize=11)
    axes[2].set_xlim(0, max(percentages) * 1.15)
    
    plt.suptitle('Reddit Sentiment Analysis: Methane & Dairy Discourse', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # --- Additional Visualization: Score vs Sentiment ---
    if 'score' in df.columns:
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot of Reddit scores by sentiment
        sentiment_order = ['Negative', 'Neutral', 'Positive']
        available_sentiments = [s for s in sentiment_order if s in df['predicted_sentiment'].values]
        
        box_data = [df[df['predicted_sentiment'] == s]['score'].values for s in available_sentiments]
        bp = axes2[0].boxplot(box_data, labels=available_sentiments, patch_artist=True)
        
        # Color the boxes
        for patch, sentiment in zip(bp['boxes'], available_sentiments):
            patch.set_facecolor(color_map.get(sentiment, '#999999'))
            patch.set_alpha(0.7)
        
        axes2[0].set_xlabel('Predicted Sentiment', fontsize=12)
        axes2[0].set_ylabel('Reddit Score (Upvotes)', fontsize=12)
        axes2[0].set_title('Reddit Post Scores by Sentiment Category', fontsize=14, fontweight='bold')
        axes2[0].grid(axis='y', alpha=0.3)
        
        # Confidence distribution by sentiment
        for sentiment in available_sentiments:
            subset = df[df['predicted_sentiment'] == sentiment]['confidence']
            axes2[1].hist(subset, bins=20, alpha=0.6, label=sentiment,
                          color=color_map.get(sentiment, '#999999'), edgecolor='#333333')
        axes2[1].set_xlabel('Confidence Score', fontsize=12)
        axes2[1].set_ylabel('Number of Posts', fontsize=12)
        axes2[1].set_title('Model Confidence by Sentiment', fontsize=14, fontweight='bold')
        axes2[1].legend(fontsize=11)
        axes2[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("✅ Final Analysis Visualizations Generated.")

# ======================================================
# PART 5: TEMPORAL SENTIMENT ANALYSIS
# ======================================================
def step_5_temporal_analysis(df):
    print("\n--- STEP 5: TEMPORAL SENTIMENT ANALYSIS ---")
    
    if 'created_utc' not in df.columns or 'predicted_sentiment' not in df.columns:
        print("⚠️ Missing timestamp or sentiment data. Skipping temporal analysis.")
        return
    
    df = df.copy()
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df['year_month'] = df['created_utc'].dt.to_period('M')
    
    earliest = df['created_utc'].min()
    latest = df['created_utc'].max()
    span_days = (latest - earliest).days
    print(f"Temporal range: {earliest.strftime('%Y-%m-%d')} → {latest.strftime('%Y-%m-%d')} ({span_days} days)")
    
    # --- 1. Monthly sentiment counts ---
    monthly = df.groupby(['year_month', 'predicted_sentiment']).size().unstack(fill_value=0)
    # Ensure all sentiment columns exist
    for col in ['Negative', 'Neutral', 'Positive']:
        if col not in monthly.columns:
            monthly[col] = 0
    monthly = monthly[['Negative', 'Neutral', 'Positive']].sort_index()
    
    # Convert period index to timestamps for plotting
    monthly.index = monthly.index.to_timestamp()
    
    # --- 2. Monthly proportions ---
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
    
    color_map = {'Negative': '#ff6b6b', 'Neutral': '#ffd93d', 'Positive': '#6bcb77'}
    
    # ===== FIGURE 1: Sentiment trend lines (counts + proportions) =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Panel A: Absolute counts with trend lines
    for sentiment in ['Negative', 'Neutral', 'Positive']:
        axes[0].plot(monthly.index, monthly[sentiment],
                     marker='o', markersize=4, linewidth=2,
                     color=color_map[sentiment], label=sentiment)
    axes[0].set_ylabel('Number of Posts', fontsize=12)
    axes[0].set_title('Monthly Sentiment Volume Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper left')
    axes[0].grid(axis='both', alpha=0.3)
    
    # Panel B: Proportion over time
    for sentiment in ['Negative', 'Neutral', 'Positive']:
        axes[1].plot(monthly_pct.index, monthly_pct[sentiment],
                     marker='o', markersize=4, linewidth=2,
                     color=color_map[sentiment], label=sentiment)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_title('Monthly Sentiment Proportion Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='upper left')
    axes[1].grid(axis='both', alpha=0.3)
    axes[1].set_ylim(0, 100)
    
    # Format x-axis dates
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(monthly) // 12)))
    plt.xticks(rotation=45, ha='right')
    
    plt.suptitle('Temporal Sentiment Evolution: Methane & Dairy Discourse',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
    
    # ===== FIGURE 2: Stacked area chart =====
    fig2, ax = plt.subplots(figsize=(14, 6))
    
    ax.stackplot(monthly_pct.index,
                 monthly_pct['Negative'], monthly_pct['Neutral'], monthly_pct['Positive'],
                 labels=['Negative', 'Neutral', 'Positive'],
                 colors=[color_map['Negative'], color_map['Neutral'], color_map['Positive']],
                 alpha=0.8)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Stacked Sentiment Proportion Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(monthly) // 12)))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # ===== FIGURE 3: Rolling average (7-day) of sentiment score =====
    # Encode sentiment as numeric: Negative=-1, Neutral=0, Positive=1
    sentiment_numeric = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df['sentiment_score'] = df['predicted_sentiment'].map(sentiment_numeric)
    df_sorted = df.sort_values('created_utc')
    
    # Compute daily mean sentiment score
    daily_score = df_sorted.set_index('created_utc').resample('D')['sentiment_score'].mean().dropna()
    
    if len(daily_score) > 7:
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        
        ax3.plot(daily_score.index, daily_score.values, alpha=0.25, color='#666666', linewidth=0.8, label='Daily')
        rolling = daily_score.rolling(window=7, min_periods=1).mean()
        ax3.plot(rolling.index, rolling.values, color='#4d96ff', linewidth=2.5, label='7-Day Rolling Avg')
        
        ax3.axhline(y=0, color='#999999', linestyle='--', linewidth=1, alpha=0.6)
        ax3.fill_between(rolling.index, rolling.values, 0,
                         where=(rolling.values >= 0), color='#6bcb77', alpha=0.15)
        ax3.fill_between(rolling.index, rolling.values, 0,
                         where=(rolling.values < 0), color='#ff6b6b', alpha=0.15)
        
        ax3.set_ylabel('Sentiment Score (-1 to +1)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title('Rolling Average Sentiment Score Over Time', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(axis='both', alpha=0.3)
        ax3.set_ylim(-1.1, 1.1)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    # Print summary statistics
    print("\nTEMPORAL SUMMARY:")
    print(f"  Posts per month (avg): {monthly.sum(axis=1).mean():.0f}")
    print(f"  Most active month: {monthly.sum(axis=1).idxmax().strftime('%B %Y')} ({monthly.sum(axis=1).max()} posts)")
    print(f"  Overall sentiment trend:")
    # Compare first half vs second half
    midpoint = monthly.index[len(monthly) // 2]
    first_half = monthly_pct.loc[monthly_pct.index < midpoint].mean()
    second_half = monthly_pct.loc[monthly_pct.index >= midpoint].mean()
    for sent in ['Negative', 'Neutral', 'Positive']:
        delta = second_half[sent] - first_half[sent]
        direction = '↑' if delta > 0 else '↓'
        print(f"    {sent}: {first_half[sent]:.1f}% → {second_half[sent]:.1f}% ({direction}{abs(delta):.1f}pp)")
    
    print("✅ Temporal Analysis Complete.")

# ======================================================
# MAIN EXECUTION FLOW
# ======================================================
if __name__ == "__main__":
    # 1. Scrape
    reddit_df = step_1_scrape_reddit()
    
    # 2. Load Pretrained Model
    sentiment_model, tokenizer, config = step_2_load_model()
    
    # 3. Map Keywords
    step_3_network_analysis(reddit_df)
    
    # 4. Final Classify
    step_4_final_analysis(reddit_df, sentiment_model, tokenizer, config)
    
    # 5. Temporal Analysis
    step_5_temporal_analysis(reddit_df)