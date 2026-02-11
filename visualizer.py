import matplotlib.pyplot as plt
import numpy as np

def visualize_sentiment_results():
    """Visualize sentiment analysis results from BERT classification."""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 12))
    
    # ============ 1. Sentiment Distribution (Pie Chart) ============
    ax1 = fig.add_subplot(2, 2, 1)
    sentiments = ['Pro', 'News', 'Neutral', 'Anti']
    counts = [137, 85, 9, 1]
    colors = ['#2ecc71', '#3498db', '#95a5a6', '#e74c3c']
    explode = (0.05, 0, 0, 0)
    
    wedges, texts, autotexts = ax1.pie(
        counts, 
        labels=sentiments, 
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        shadow=True
    )
    ax1.set_title('Reddit Post Sentiment Distribution\n(232 posts analyzed)', fontsize=14, fontweight='bold')
    
    # ============ 2. Sentiment Counts (Bar Chart) ============
    ax2 = fig.add_subplot(2, 2, 2)
    bars = ax2.bar(sentiments, counts, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Sentiment Category', fontsize=12)
    ax2.set_ylabel('Number of Posts', fontsize=12)
    ax2.set_title('Sentiment Classification Counts', fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylim(0, max(counts) * 1.15)
    
    # ============ 3. Classification Report Metrics ============
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Metrics from classification report
    categories = ['Anti', 'Neutral', 'Pro', 'News']
    precision = [0.70, 0.70, 0.83, 0.81]
    recall = [0.65, 0.56, 0.87, 0.87]
    f1_score = [0.67, 0.62, 0.85, 0.84]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax3.bar(x - width, precision, width, label='Precision', color='#9b59b6', edgecolor='black')
    bars2 = ax3.bar(x, recall, width, label='Recall', color='#f39c12', edgecolor='black')
    bars3 = ax3.bar(x + width, f1_score, width, label='F1-Score', color='#1abc9c', edgecolor='black')
    
    ax3.set_xlabel('Sentiment Category', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('BERT Model Performance by Category\n(Test Accuracy: 79%)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='lower right')
    ax3.set_ylim(0, 1.0)
    ax3.axhline(y=0.79, color='red', linestyle='--', alpha=0.7, label='Overall Accuracy')
    
    # ============ 4. Training Loss Over Epochs ============
    ax4 = fig.add_subplot(2, 2, 4)
    epochs = [1, 2, 3]
    avg_loss = [0.6985, 0.4168, 0.2535]
    
    ax4.plot(epochs, avg_loss, 'o-', color='#e74c3c', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax4.fill_between(epochs, avg_loss, alpha=0.3, color='#e74c3c')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Average Loss', fontsize=12)
    ax4.set_title('BERT Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax4.set_xticks(epochs)
    ax4.set_ylim(0, max(avg_loss) * 1.1)
    
    # Add loss values as annotations
    for epoch, loss in zip(epochs, avg_loss):
        ax4.annotate(f'{loss:.4f}', (epoch, loss), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    # ============ Final Adjustments ============
    plt.suptitle('Climate Sentiment Analysis: Methane & Dairy Industry\n(Reddit Data)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to 'sentiment_analysis_results.png'")


if __name__ == "__main__":
    visualize_sentiment_results()
