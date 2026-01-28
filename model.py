import praw
import pandas as pd
import re
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
        
        keywords = 'methane cows environment climate greenhouse gas emission dairy industry'
        limit = 500
        print(f"Searching Reddit for: '{keywords}'...")
        
        data = []
        # We search 'all' subreddits to catch r/farming, r/climate, r/science
        for post in reddit.subreddit('all').search(keywords, limit=limit):
            data.append({
                'text': f"{post.title} {post.selftext}",
                'score': post.score,
                'id': post.id
            })
            
        df = pd.DataFrame(data)
        print(f"✅ Success! Scraped {len(df)} posts.")
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
        ] * 10
        return pd.DataFrame({'text': mock_data, 'score': [10]*80})

# ======================================================
# PART 2: BUILD CLASSIFICATION MODEL (Kaggle Dataset)
# ======================================================
def step_2_build_model():
    print("\n--- STEP 2: BUILDING SENTIMENT MODEL ---")
    
    # 1. Load Data
    # load kaggle_climate_data.csv
    print("Loading Kaggle climate sentiment data...")

    kaggle_df = pd.read_csv('kaggle_climate_data.csv') # Ensure this file is in the same directory
    print(f"Loaded {len(kaggle_df)} rows from Kaggle dataset.")

    # Map sentiments to numerical labels
    # CSV labels:
    #   -1 (Anti): does not believe in man-made climate change
    #    0 (Neutral): neither supports nor refutes man-made climate change
    #    1 (Pro): supports the belief of man-made climate change
    #    2 (News): links to factual news about climate change
    sentiment_map = {-1: 0, 0: 1, 1: 2, 2: 3}
    kaggle_df['sentiment_label'] = kaggle_df['sentiment'].map(sentiment_map)
    train_texts = kaggle_df['message'].tolist()  # Column is 'message', not 'text'
    train_labels = kaggle_df['sentiment_label'].tolist()
    
    df_train = pd.DataFrame({'text': train_texts, 'sentiment': train_labels})
    
    # 2. Preprocessing & Vectorization
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df_train['text'])
    y = df_train['sentiment']
    
    # 3. Train Model (Logistic Regression)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    print("✅ Model Trained.")
    print(f"Accuracy on test set: {model.score(X_test, y_test):.2f}")
    
    # Show detailed classification metrics
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Anti', 'Neutral', 'Pro', 'News']))
    
    return model, vectorizer

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
    
    # 2. Extract co-occurrences
    co_occurrence = Counter()
    
    for text in df['text']:
        # Simple cleaning: remove punctuation, lowercase
        clean = re.sub(r'[^\w\s]', '', text.lower())
        # Filter tokens: no stop words, must be longer than 2 chars (keeps "cow")
        tokens = [w for w in clean.split() if w not in stop_words and len(w) > 2]
        
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
        cmap=plt.cm.YlGnBu, # A nice Blue-Green-Yellow colormap
        alpha=0.9,
        edgecolors='#333333', # Thin dark border for definition
        linewidths=1.0
    )
    
    # 3. Edge Sizing & Coloring based on 'weight' (strength)
    #    Scale the width so it's visible but not overwhelming.
    edge_widths = [G[u][v]['weight'] * 0.3 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
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
def step_4_final_analysis(df, model, vectorizer):
    print("\n--- STEP 4: FINAL CLASSIFICATION ---")
    
    # 1. Transform scraped data using the SAME vectorizer from Step 2
    X_new = vectorizer.transform(df['text'])
    
    # 2. Predict
    predictions = model.predict(X_new)
    
    # Map back to labels
    label_map = {0: 'Anti', 1: 'Neutral', 2: 'Pro', 3: 'News'}
    df['predicted_sentiment'] = [label_map[p] for p in predictions]
    
    # 3. Results
    print("\nFINAL RESULTS SUMMARY:")
    print(df['predicted_sentiment'].value_counts())
    
    print("\nSAMPLE CLASSIFICATIONS:")
    print(df[['text', 'predicted_sentiment']].head(5))

# ======================================================
# MAIN EXECUTION FLOW
# ======================================================
if __name__ == "__main__":
    # 1. Scrape
    reddit_df = step_1_scrape_reddit()
    
    # 2. Build Model
    sentiment_model, tfidf_vectorizer = step_2_build_model()
    
    # 3. Map Keywords
    step_3_network_analysis(reddit_df)
    
    # 4. Final Classify
    step_4_final_analysis(reddit_df, sentiment_model, tfidf_vectorizer)