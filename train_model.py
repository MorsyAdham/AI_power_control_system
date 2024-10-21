import pandas as pd
from ai_model import PowerControlAIModel
from sklearn.model_selection import train_test_split

# Step 1: Load the Dataset
df = pd.read_csv("balanced_appliance_commands_dataset.csv")

# Step 2: Prepare the Dataset
texts = df['text'].tolist()
labels = df['label'].tolist()

# Split dataset for training and evaluation
train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 3: Initialize the AI Model
model = PowerControlAIModel(num_labels=len(set(labels)))  # Set number of labels dynamically

# Step 4: Create Datasets
train_dataset = model.create_dataset(train_texts, train_labels)
eval_dataset = model.create_dataset(eval_texts, eval_labels)

# Step 5: Train the Model
model.train(train_dataset=train_dataset, eval_dataset=eval_dataset, output_dir="./saved_model")
