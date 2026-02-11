import praw
import pandas as pd
import re
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
        
        target = 2000
        seen_ids = set()
        data = []
        
        print(f"Searching Reddit with multiple queries to reach {target} posts...")
        
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
                                if post.id not in seen_ids:
                                    seen_ids.add(post.id)
                                    data.append({
                                        'text': f"{post.title} {post.selftext}",
                                        'score': post.score,
                                        'id': post.id
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
        return pd.DataFrame({'text': mock_data, 'score': [10]*1000})

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
    #    We add more domain-specific "noise" words to get a clearer picture.
    stop_words = set(stopwords.words('english'))
    additional_stops = {
        'methane', 'dairy', 'http', 'https', 'cows', 'industry', 
        'would', 'could', 'should', 'people', 'like', 'make', 'think'
    }
    stop_words.update(additional_stops)
    lemmatizer = WordNetLemmatizer()
    
    # 2. Extract co-occurrences
    co_occurrence = Counter()
    
    for text in df['text']:
        # Simple cleaning: remove punctuation, lowercase
        clean = re.sub(r'[^\w\s]', '', text.lower())
        # Filter tokens: no stop words, must be longer than 2 chars (keeps "cow")
        # Lemmatize to reduce words to base form (e.g., "emissions" -> "emission")
        tokens = [lemmatizer.lemmatize(w) for w in clean.split() if w not in stop_words and len(w) > 2]
        
        # Create edges between all unique words in the same post
        unique_tokens = sorted(list(set(tokens)))
        for i in range(len(unique_tokens)):
            for j in range(i + 1, len(unique_tokens)):
                edge = (unique_tokens[i], unique_tokens[j])
                co_occurrence[edge] += 1

    # 3. Build Graph with NetworkX
    G = nx.Graph()
    
    # Add top 80 strongest edges.
    # IMPORTANT: We filter out edges with weight 1 to remove clutter.
    num_edges_added = 0
    for (w1, w2), weight in co_occurrence.most_common(150):
        if weight > 1: 
            G.add_edge(w1, w2, weight=weight)
            num_edges_added += 1
        if num_edges_added >= 80:
            break
            
    print(f"Network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if G.number_of_nodes() == 0:
        print("⚠️ Not enough significant co-occurrences to plot.")
        return
    
    # ===================================================
    # IMPROVED VISUALIZATION CODE
    # ===================================================
    # Set up the figure with a clean white background
    plt.figure(figsize=(12, 10), facecolor='white')
    
    # 1. Layout: 'k' controls spacing. Larger k = more spread out.
    #    'seed' ensures the same layout every time you run it.
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # 2. Node Sizing & Coloring based on "Degree" (importance)
    degrees = dict(G.degree())
    # Scale factor to make nodes a good size
    node_sizes = [v * 200 for v in degrees.values()]
    # Use degree count for color mapping
    node_colors = [v for v in degrees.values()]

    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Pastel1, # Light pastel colormap
        alpha=0.9,
        edgecolors='#333333', # Thin dark border for definition
        linewidths=1.0
    )
    
    # 3. Draw Edges
    nx.draw_networkx_edges(
        G, pos,
        width=1.0,
        edge_color='#999999', # Solid medium grey
        alpha=0.6 # Slight transparency
    )
    
    # 4. Labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family='sans-serif',
        font_weight='bold',
        font_color='#222222' # Dark grey for high contrast
    )
    
    plt.title("Keyword Co-occurrence: Methane & Dairy Discourse", fontsize=14, fontweight='bold', color='#222222')
    plt.axis('off') # Turn off the axis for a clean look
    plt.tight_layout()
    plt.show() # This will open the new, improved graph window
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