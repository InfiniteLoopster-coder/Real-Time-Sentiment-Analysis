from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import mlflow
import mlflow.pytorch

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model():
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # e.g., 0: negative, 1: neutral, 2: positive
    
    # Dummy dataset for demonstration (replace with actual tweet data and labels)
    texts = ["I love this!", "This is terrible.", "I feel okay."]
    labels = [2, 0, 1]
    dataset = TweetDataset(texts, labels, tokenizer)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_dir='./logs',
        logging_steps=10
    )
    
    # Start MLflow experiment tracking
    mlflow.set_experiment("Sentiment Analysis Experiment")
    with mlflow.start_run():
        mlflow.log_param("num_epochs", 3)
        mlflow.log_param("batch_size", 2)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        
        trainer.train()
        
        # Save the model locally
        model.save_pretrained("./sentiment_model")
        tokenizer.save_pretrained("./sentiment_model")
        
        # Log the model with MLflow
        mlflow.pytorch.log_model(model, "model")
        print("Model training complete and logged with MLflow.")

if __name__ == "__main__":
    train_model()
