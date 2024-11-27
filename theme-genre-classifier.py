import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, hamming_loss
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieGenreDataset(Dataset):
    """Custom Dataset for movie genre classification"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
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
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.FloatTensor(label),
        }


class MovieGenreClassifier:
    """Movie Genre Classification Model"""

    def __init__(self, model_name="bert-base-uncased", num_labels=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlb = MultiLabelBinarizer()
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        logger.info(f"Using device: {self.device}")

    def prepare_data(self, df, test_size=0.2):
        """Prepare and split data"""
        # Convert genre strings to lists
        genres = df["genre"].str.split("|")

        # Transform labels
        genre_labels = self.mlb.fit_transform(genres)

        # Update num_labels if not set
        if self.num_labels is None:
            self.num_labels = len(self.mlb.classes_)
            logger.info(f"Number of genres: {self.num_labels}")

            # Initialize model after getting num_labels
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="multi_label_classification",
            ).to(self.device)

        # Split data
        texts = df["description"].values
        X_train, X_test, y_train, y_test = train_test_split(
            texts, genre_labels, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        train_texts,
        train_labels,
        val_texts=None,
        val_labels=None,
        batch_size=8,
        epochs=3,
        learning_rate=2e-5,
    ):
        """Train the model"""
        # Create datasets
        train_dataset = MovieGenreDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_texts is not None and val_labels is not None:
            val_dataset = MovieGenreDataset(val_texts, val_labels, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Average training loss: {avg_train_loss:.4f}"
            )

            # Validation
            if val_texts is not None and val_labels is not None:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Validation loss: {val_loss:.4f}")

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()

        return total_loss / len(dataloader)

    def predict(self, texts, threshold=0.5):
        """Make predictions"""
        self.model.eval()
        dataset = MovieGenreDataset(
            texts, np.zeros((len(texts), self.num_labels)), self.tokenizer
        )
        dataloader = DataLoader(dataset, batch_size=8)
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                probs = torch.sigmoid(logits)
                predictions.extend(probs.cpu().numpy())

        predictions = np.array(predictions)
        return (predictions >= threshold).astype(int)

    def get_metrics(self, true_labels, predicted_labels):
        """Calculate performance metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average="weighted"
        )
        h_loss = hamming_loss(true_labels, predicted_labels)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "hamming_loss": h_loss,
        }

        return metrics

    def analyze_predictions(self, texts, true_labels, predicted_labels, n_samples=5):
        """Analyze prediction examples"""
        genre_names = self.mlb.classes_
        analysis = {"good_predictions": [], "bad_predictions": []}

        for i in range(len(texts)):
            true_genres = [
                genre_names[j]
                for j in range(len(true_labels[i]))
                if true_labels[i][j] == 1
            ]
            pred_genres = [
                genre_names[j]
                for j in range(len(predicted_labels[i]))
                if predicted_labels[i][j] == 1
            ]

            if set(true_genres) == set(pred_genres):
                if len(analysis["good_predictions"]) < n_samples:
                    analysis["good_predictions"].append(
                        {
                            "text": texts[i],
                            "true_genres": true_genres,
                            "predicted_genres": pred_genres,
                        }
                    )
            else:
                if len(analysis["bad_predictions"]) < n_samples:
                    analysis["bad_predictions"].append(
                        {
                            "text": texts[i],
                            "true_genres": true_genres,
                            "predicted_genres": pred_genres,
                        }
                    )

        return analysis


def main():
    # Load synthetic data
    logger.info("Loading synthetic data...")
    df = pd.read_csv("data/real_data.csv")

    # Initialize classifier
    classifier = MovieGenreClassifier()

    # Prepare data
    logger.info("Preparing data...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)

    # Train model
    logger.info("Training model...")
    classifier.train_model(X_train, y_train, X_test, y_test)

    # Make predictions
    logger.info("Making predictions...")
    predictions = classifier.predict(X_test)

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = classifier.get_metrics(y_test, predictions)
    logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")

    # Analyze predictions
    logger.info("Analyzing predictions...")
    analysis = classifier.analyze_predictions(X_test, y_test, predictions)

    # Save results
    with open("results/real_results.json", "w") as f:
        json.dump({"metrics": metrics, "analysis": analysis}, f, indent=2)


if __name__ == "__main__":
    main()
