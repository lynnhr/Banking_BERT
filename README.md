# Model Training_Banking dataset with Labels
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

1- Import important libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

2- Check for null values
```python
dataset.info()
dataset.isnull().sum()
```
![Image](https://github.com/user-attachments/assets/b985ce71-3ce0-417e-ae47-97d1010b85d5)

3- Split the dataset into x (features) and y (dependent variable)
```python
dataset=pd.read_csv('train.csv')
x=dataset.iloc[:, :-1].values # : range select all rows
y=dataset.iloc[:,-1].values   # -1 means the last row
```

4- Encode Dependent Variable
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(y)
print(Y)
```

## Bernouilli (82%)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import nltk

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a string
    return ' '.join(tokens)

# Apply preprocessing to the text data
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Convert text data into numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer(binary=True, stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Encode the target variable (labels)
y_train = train_df['category']
y_test = test_df['category']

# Train a Bernoulli Naive Bayes model
bnb = BernoulliNB(alpha=0.1)  # Using a fixed alpha value for simplicity
bnb.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = bnb.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

## GridSearch Logistic regression (86%)
```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load the training and testing datasets
train_df = pd.read_csv('train.csv')  # Replace with your training data path
test_df = pd.read_csv('test.csv')    # Replace with your testing data path

# Display the first few rows of the datasets
print("Training Data:")
print(train_df.head())

print("\nTesting Data:")
print(test_df.head())

# Check for missing values
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test_df.isnull().sum())

# Drop rows with missing values (if any)
train_df = train_df.dropna()
test_df = test_df.dropna()

# Encode the target variable (labels)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_test = label_encoder.transform(test_df['category'])

# Create a pipeline with TfidfVectorizer and LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(binary=True, stop_words='english', ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))  # Increase max_iter for convergence
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'tfidf__max_features': [5000, 10000],  # Experiment with more features
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Try unigrams and bigrams
    'clf__C': [0.1, 1, 10],  # Regularization strength
    'clf__penalty': ['l2'],  # L2 regularization is common for Logistic Regression
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(train_df['text'], y_train)

# Print the best parameters and accuracy
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Predict the labels for the test set
y_pred = grid_search.predict(test_df['text'])

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model (optional)
import joblib
joblib.dump(grid_search, 'text_classification_model.pkl')
```

## RandomFOrest (79%)
```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the training and testing datasets
train_df = pd.read_csv('train.csv')  # Replace with your training data path
test_df = pd.read_csv('test.csv')    # Replace with your testing data path

# Display the first few rows of the datasets
print("Training Data:")
print(train_df.head())

print("\nTesting Data:")
print(test_df.head())

# Check for missing values
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test_df.isnull().sum())

# Drop rows with missing values (if any)
train_df = train_df.dropna()
test_df = test_df.dropna()

# Encode the target variable (labels)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_test = label_encoder.transform(test_df['category'])

# Create a pipeline with TfidfVectorizer and Random Forest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)),  # Use more features
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42))  # Tune these parameters
])

# Train the model
pipeline.fit(train_df['text'], y_train)

# Predict the labels for the test set
y_pred = pipeline.predict(test_df['text'])

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

```

## Bert
```python
# Install required libraries
!pip install transformers datasets accelerate

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the training and testing datasets
train_df = pd.read_csv('train.csv')  # Replace with your training data path
test_df = pd.read_csv('test.csv')    # Replace with your testing data path

# Display the first few rows of the datasets
print("Training Data:")
print(train_df.head())

print("\nTesting Data:")
print(test_df.head())

# Check for missing values
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test_df.isnull().sum())

# Drop rows with missing values (if any)
train_df = train_df.dropna()
test_df = test_df.dropna()

# Encode the target variable (labels)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_test = label_encoder.transform(test_df['category'])

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data with reduced max_length
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=64)

# Convert to PyTorch datasets
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}  # Move data to GPU
        item['labels'] = torch.tensor(self.labels[idx]).to(device)  # Move labels to GPU
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, y_train)
test_dataset = TextDataset(test_encodings, y_test)

# Load BERT model and move it to GPU
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)

# Define training arguments with GPU and mixed precision
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,                # Log every 10 steps
    evaluation_strategy="epoch",     # Evaluate after each epoch
    save_strategy="epoch",           # Save model after each epoch
    load_best_model_at_end=True,     # Load the best model at the end
    report_to="none",                # Disable wandb
    fp16=True,                       # Enable mixed precision
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```
download
```python
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_tokenizer')
!zip -r saved_model.zip ./saved_model
!zip -r saved_tokenizer.zip ./saved_tokenizer
from google.colab import files

# Download the model
files.download('saved_model.zip')

# Download the tokenizer
#files.download('saved_tokenizer.zip')
```

```python
import pickle

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
```
visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the trainer object is available
if 'trainer' in globals():
    print("Trainer object found. Proceeding to visualize metrics.")
else:
    print("Trainer object not found. Please ensure the model has been trained.")

# Extract training history
history = trainer.state.log_history

# Extract training and evaluation metrics
train_loss = [entry['loss'] for entry in history if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in history if 'eval_loss' in entry]

# Calculate the number of steps per epoch
steps_per_epoch = len(train_loss) // len(eval_loss)

# Create x-axis values for training loss (steps)
train_steps = list(range(len(train_loss)))

# Create x-axis values for evaluation loss (epochs)
eval_steps = [steps_per_epoch * (i + 1) for i in range(len(eval_loss))]

# Plot training and evaluation loss
plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_loss, label='Training Loss', marker='', linestyle='-')
plt.plot(eval_steps, eval_loss, label='Evaluation Loss', marker='o', linestyle='--')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Over Steps')
plt.legend()
plt.grid(True)
plt.show()

# Plot evaluation accuracy (if available)
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
```
learning rate
```python
import matplotlib.pyplot as plt
import seaborn as sns
history = trainer.state.log_history
# Extract learning rates from the logs
learning_rates = [entry['learning_rate'] for entry in history if 'learning_rate' in entry]

# Plot learning rate schedule
plt.figure(figsize=(10, 5))
plt.plot(range(len(learning_rates)), learning_rates, label='Learning Rate', marker='')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule Over Steps')
plt.legend()
plt.grid(True)
plt.show()
```



