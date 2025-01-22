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
```




