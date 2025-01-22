# Bernouilli
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
2- Split the dataset into x (features) and y (dependent variable)
```python
dataset=pd.read_csv('train.csv')
x=dataset.iloc[:, :-1].values # : range select all rows
y=dataset.iloc[:,-1].values   # -1 means the last row
```
3- Check for null values
```python
dataset.info()
dataset.isnull().sum()
```
![Image](https://github.com/user-attachments/assets/b985ce71-3ce0-417e-ae47-97d1010b85d5)

4- Encoding the independent variable
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#create object of class()
#transformer takes 3 arguments: kind of transformation(here encoding),what kind of encoding ,index of columns
#remainder what we don't want to encode
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])],remainder="passthrough")    
# np.array ensures that the result is converted into a NumPy array
X=np.array(ct.fit_transform(x)) #X the new x updated
print(X)
```
![Image](https://github.com/user-attachments/assets/134f8b04-db5f-45ca-9840-b2762dcb8925)

5- Encode Dependent Variable
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(y)
print(Y)
```


