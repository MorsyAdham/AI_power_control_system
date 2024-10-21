import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """Custom dataset class for handling text and labels."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class PowerControlAIModel:
    def __init__(self, num_labels=4):
        self.model_name = "bert-base-uncased"
        self.num_labels = num_labels
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)

    def create_dataset(self, texts, labels):
        # Create a SimpleDataset object
        return SimpleDataset(texts, labels, self.tokenizer)

    def train(self, train_dataset, eval_dataset, output_dir="./saved_model", epochs=3, batch_size=8):
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train the model
        training_results = trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Visualize training metrics
        self.visualize_training_metrics(trainer, output_dir)
        print("Model training completed and saved to", output_dir)

    def visualize_training_metrics(self, trainer, output_dir="./saved_model/plots"):
        """Visualizes training metrics such as loss and accuracy."""
        # Make the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract training history
        metrics = trainer.state.log_history
        train_loss = [m["loss"] for m in metrics if "loss" in m]
        eval_loss = [m["eval_loss"] for m in metrics if "eval_loss" in m]
        train_accuracy = [m["eval_accuracy"] for m in metrics if "eval_accuracy" in m]
        steps = range(1, len(train_loss) + 1)

        # Plot Loss
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_loss, label="Training Loss", color="blue")
        plt.plot(range(1, len(eval_loss) + 1), eval_loss, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
        plt.close()

        # Plot Accuracy
        if train_accuracy:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label="Validation Accuracy", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Validation Accuracy")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
            plt.close()

        # Plot Learning Rate
        learning_rates = [m["learning_rate"] for m in metrics if "learning_rate" in m]
        if learning_rates:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(learning_rates) + 1), learning_rates, label="Learning Rate", color="purple")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, "learning_rate_plot.png"))
            plt.close()

    def load_model(self, model_dir="./saved_model"):
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        print("Model loaded from", model_dir)

    def predict(self, command):
        self.model.eval()
        inputs = self.tokenizer(command, return_tensors="pt", padding="max_length", truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        intents = ["Turn on appliance", "Turn off appliance", "Set timer for appliance", "Map appliance to port"]
        return intents[predicted_label]
