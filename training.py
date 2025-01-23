
!pip install transformers datasets accelerate

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


train_df = pd.read_csv('train.csv')  
test_df = pd.read_csv('test.csv')    


print("Training Data:")
print(train_df.head())

print("\nTesting Data:")
print(test_df.head())


print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test_df.isnull().sum())


train_df = train_df.dropna()
test_df = test_df.dropna()


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_test = label_encoder.transform(test_df['category'])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=64)


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()} 
        item['labels'] = torch.tensor(self.labels[idx]).to(device)  
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, y_train)
test_dataset = TextDataset(test_encodings, y_test)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)


training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    warmup_steps=500,                
    weight_decay=0.01,              
    logging_dir='./logs',            
    logging_steps=10,                
    evaluation_strategy="epoch",     
    save_strategy="epoch",           
    load_best_model_at_end=True,    
    report_to="none",               
    fp16=True,                       
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()


predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

import matplotlib.pyplot as plt
import seaborn as sns


if 'trainer' in globals():
    print("Trainer object found. Proceeding to visualize metrics.")
else:
    print("Trainer object not found. Please ensure the model has been trained.")


history = trainer.state.log_history


train_loss = [entry['loss'] for entry in history if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in history if 'eval_loss' in entry]

steps_per_epoch = len(train_loss) // len(eval_loss)


train_steps = list(range(len(train_loss)))


eval_steps = [steps_per_epoch * (i + 1) for i in range(len(eval_loss))]

plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_loss, label='Training Loss', marker='', linestyle='-')
plt.plot(eval_steps, eval_loss, label='Evaluation Loss', marker='o', linestyle='--')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Over Steps')
plt.legend()
plt.grid(True)
plt.show()


if 'eval_accuracy' in history[0]:
    eval_accuracy = [entry['eval_accuracy'] for entry in history if 'eval_accuracy' in entry]
    plt.figure(figsize=(10, 5))
    plt.plot(eval_steps, eval_accuracy, label='Evaluation Accuracy', marker='o', color='green')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy Over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Evaluation accuracy not found in logs.")


import matplotlib.pyplot as plt
import seaborn as sns
history = trainer.state.log_history

learning_rates = [entry['learning_rate'] for entry in history if 'learning_rate' in entry]


plt.figure(figsize=(10, 5))
plt.plot(range(len(learning_rates)), learning_rates, label='Learning Rate', marker='')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule Over Steps')
plt.legend()
plt.grid(True)
plt.show()

import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_tokenizer')
!zip -r saved_model.zip ./saved_model
!zip -r saved_tokenizer.zip ./saved_tokenizer
from google.colab import files
files.download('saved_model.zip')
files.download('saved_tokenizer.zip')
