import praw
import pandas as pd
import re
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

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
        
        keywords = 'methane cows climate greenhouse gas emission dairy industry'
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
# PART 2: BERT DATASET CLASS
# ======================================================
class ClimateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ======================================================
# PART 2: BUILD CLASSIFICATION MODEL (BERT)
# ======================================================
def step_2_build_model():
    print("\n--- STEP 2: BUILDING BERT SENTIMENT MODEL ---")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading Kaggle climate sentiment data...")
    kaggle_df = pd.read_csv('kaggle_climate_data.csv')
    print(f"Loaded {len(kaggle_df)} rows from Kaggle dataset.")

    # Map sentiments to numerical labels
    # CSV labels:
    #   -1 (Anti): does not believe in man-made climate change
    #    0 (Neutral): neither supports nor refutes man-made climate change
    #    1 (Pro): supports the belief of man-made climate change
    #    2 (News): links to factual news about climate change
    sentiment_map = {-1: 0, 0: 1, 1: 2, 2: 3}
    kaggle_df['sentiment_label'] = kaggle_df['sentiment'].map(sentiment_map)
    
    texts = kaggle_df['message'].tolist()
    labels = kaggle_df['sentiment_label'].tolist()
    
    # 2. Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # 3. Initialize BERT tokenizer and model
    print("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=4  # Anti, Neutral, Pro, News
    )
    model.to(device)
    
    # 4. Create datasets and dataloaders
    train_dataset = ClimateDataset(train_texts, train_labels, tokenizer)
    test_dataset = ClimateDataset(test_texts, test_labels, tokenizer)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 5. Training setup
    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # 6. Training loop
    print(f"Training BERT for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    print("✅ BERT Model Trained.")
    
    # 7. Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Accuracy on test set: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Anti', 'Neutral', 'Pro', 'News']))
    
    return model, tokenizer, device

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
def step_4_final_analysis(df, model, tokenizer, device):
    print("\n--- STEP 4: FINAL CLASSIFICATION ---")
    
    model.eval()
    label_map = {0: 'Anti', 1: 'Neutral', 2: 'Pro', 3: 'News'}
    predictions = []
    
    # Process in batches
    batch_size = 16
    texts = df['text'].tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
            batch_texts = texts[i:i+batch_size]
            
            encoding = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    df['predicted_sentiment'] = [label_map[p] for p in predictions]
    
    # Results
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
    sentiment_model, tokenizer, device = step_2_build_model()
    
    # 3. Map Keywords
    step_3_network_analysis(reddit_df)
    
    # 4. Final Classify
    step_4_final_analysis(reddit_df, sentiment_model, tokenizer, device)