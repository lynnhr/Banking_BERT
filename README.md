# Model Training_Banking dataset with Labels
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

Model: BERT from transformers 
training time:4 hours

1- Import important libraries
```python
!pip install transformers datasets accelerate
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
```
2- Load datasets
```python
train_df = pd.read_csv('train.csv')  
test_df = pd.read_csv('test.csv')    
```
3- Print first few rows
```python
print("Training Data:")
print(train_df.head())

print("\nTesting Data:")
print(test_df.head())
```
![Image](https://github.com/user-attachments/assets/c7f1ce41-3372-4703-aeec-454f853e73fa)

4- Check for null values
```python
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test_df.isnull().sum())
```
![Image](https://github.com/user-attachments/assets/5fc3a8b6-5840-43e9-a6df-fd1be80a5afa)

5- Encode Dependent Variable
```python
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_test = label_encoder.transform(test_df['category'])
```
6-Tokenize data (independent variable)
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # load BERT tokenizer

train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=64)
```
7- Convert to Pytorch dataset, before passing the data to the model
```python
class TextDataset(Dataset):
    def __init__(self, encodings, labels): #__init__ initializes the dataset with encodings and labels
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx): # __getitem__ retrieves a single data sample at a given index
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}  # .to(device) is used when working with gpu instead of cpu (gpu is better and faster for tranformers models)
        item['labels'] = torch.tensor(self.labels[idx]).to(device)  # toech.tensor() converts them to pytorch tensors, idx is the index
        return item

    def __len__(self): 
        return len(self.labels) #returns length of labels list

train_dataset = TextDataset(train_encodings, y_train)
test_dataset = TextDataset(test_encodings, y_test)
```

8- Load BERT model
this model from transformers is used for predicting categories or labels
```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)
#'bert-base-uncased' base refers to the smallest version of the model, uncased means the tokenizer will convert the text to lowercase andd ignore case sensitivity
#num_labels=len(label_encoder.classes_) label_encoder.classes_ stores the unique labels in the dataset
```
9- Define Training arguments
```python
training_args = TrainingArguments(
    output_dir='./results',          # output directory: after training the model and logs will be saved in 'results' folder
    num_train_epochs=3,              # number of epochs: an epoch is a full pass through the training data
    per_device_train_batch_size=16,  # batch: chunk of data processed
    per_device_eval_batch_size=16,   
    warmup_steps=500,                
    weight_decay=0.01,              
    logging_dir='./logs',            # logs will be stores in this directory, a log is a messages recorded during the execution of a program
    logging_steps=10,                # log every 10 steps
    evaluation_strategy="epoch",     # evaluate after each epoch
    save_strategy="epoch",           # save model after each epoch
    load_best_model_at_end=True,     # load the best model at the end
    report_to="none",                # disable wandb, i disabled this becuase it asked to create an account in wandb and enter the API
    fp16=True,                       
)
```
10- Define trainer
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
```
11- Train
```python
trainer.train()
```
![Image](https://github.com/user-attachments/assets/23c0cd3a-6a4d-48f7-b18e-e8c0fee5b039)

12- Evaluate the model
```python
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
```
13- Calculate the accuracy 
```python
accuracy = accuracy_score(y_test, y_pred) # we calculate the accuracy by using the test and predictions
print(f"\nTest Accuracy: {accuracy:.4f}")
```
![Image](https://github.com/user-attachments/assets/b7b39c02-7967-4009-a966-c52ce6d05a88)

14- Print the classification report
```python
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```
![Image](https://github.com/user-attachments/assets/0f2760e0-2c9c-4bec-b292-284b89f18b92)
![Image](https://github.com/user-attachments/assets/7bd1c861-7541-411e-b930-02df5195442e)

15- Visualize Training and Evaluation Loss
    using log history
```python
import matplotlib.pyplot as plt
import seaborn as sns

history = trainer.state.log_history #extract training history

# extract training and evaluation metrics
train_loss = [entry['loss'] for entry in history if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in history if 'eval_loss' in entry]

# calculate the number of steps per epoch
steps_per_epoch = len(train_loss) // len(eval_loss)

# create x axis
train_steps = list(range(len(train_loss))) #generates a list of integers 0,1,2... corresponding to each training step.
eval_steps = [steps_per_epoch * (i + 1) for i in range(len(eval_loss))]

# plot
plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_loss, label='Training Loss', marker='', linestyle='-') #y axis train loss
plt.plot(eval_steps, eval_loss, label='Evaluation Loss', marker='o', linestyle='--')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Over Steps')
plt.legend()
plt.grid(True)
plt.show()
```
![Image](https://github.com/user-attachments/assets/99846326-5617-42f7-9741-84131a14e05d)
16- Visualize Learning Rat
```python
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
```
![Image](https://github.com/user-attachments/assets/9e27f630-84de-412c-80b7-290c0ffa6f40)

17- Save thye model+ tokenizer+ label_encoder
    this model is most suitable for tranformers models
```python
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_tokenizer')

#zip the files before downloading them
!zip -r saved_model.zip ./saved_model
!zip -r saved_tokenizer.zip ./saved_tokenizer

from google.colab import files
files.download('saved_model.zip')
files.download('saved_tokenizer.zip')

#use .pkl for label_encoder
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
```

18- Load the trained  model in pycharm to test and predict 
```python
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
import pickle

model = BertForSequenceClassification.from_pretrained('saved_model')
tokenizer = BertTokenizer.from_pretrained('saved_tokenizer')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# print the class names
print("Label Mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i}: {class_name}")


def predict(text):
    # tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)

    
    with torch.no_grad():
        outputs = model(**inputs)

    
    logits = outputs.logits  #higher logits generally correspond to higher confidence for that class.
    predicted_class = torch.argmax(logits, dim=-1).item()  #torch.argmax(logits, dim=-1) identifies the index of the highest value   .item():converts tensor result into a python scalar(example:0,1..)

    return predicted_class


sample_text = "i want to change my card"
predicted_label = predict(sample_text)

print(f"Predicted Label: {predicted_label}")
print(f"Predicted Class: {label_encoder.classes_[predicted_label]}")
```
![Image](https://github.com/user-attachments/assets/dc2b1706-96b5-43ca-82f5-ead4320226d6)

#Performed Traning with other models but accuracy<90%

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


lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)


vectorizer = TfidfVectorizer(binary=True, stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])


y_train = train_df['category']
y_test = test_df['category']


bnb = BernoulliNB(alpha=0.1)  
bnb.fit(X_train, y_train)


y_pred = bnb.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

## GridSearch Logistic regression (86%)
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


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


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(binary=True, stop_words='english', ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))  # Increase max_iter for convergence
])


param_grid = {
    'tfidf__max_features': [5000, 10000],  # Experiment with more features
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Try unigrams and bigrams
    'clf__C': [0.1, 1, 10],  # Regularization strength
    'clf__penalty': ['l2'],  # L2 regularization is common for Logistic Regression
}


grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(train_df['text'], y_train)


print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")


y_pred = grid_search.predict(test_df['text'])


accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

import joblib
joblib.dump(grid_search, 'text_classification_model.pkl')
```

## RandomForest (79%)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


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


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)), 
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42))  
])


pipeline.fit(train_df['text'], y_train)


y_pred = pipeline.predict(test_df['text'])


accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

```


