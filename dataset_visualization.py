import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Step 1: Load the dataset
file_path = "balanced_appliance_commands_dataset.csv"
df = pd.read_csv(file_path)

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Step 2: Plot the class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Class Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/class_distribution.png')
plt.show()

# Step 3: Plot text length distribution
df['text_length'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(df['text_length'], bins=20, kde=True, color='blue')
plt.title('Text Length Distribution')
plt.xlabel('Number of Words in Text')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/text_length_distribution.png')
plt.show()

# Step 4: Generate a word cloud
text_combined = ' '.join(df['text'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Commands')
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/word_cloud.png')
plt.show()

print("Visualizations saved in the 'plots' folder.")
