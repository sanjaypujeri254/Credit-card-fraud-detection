import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the data 
data = pd.read_csv("C:\\Users\\HP\\Desktop\\DATA SCIENCE\\Credit card fraud\\creditcard.csv")
print(data.head())
print(data.tail())
# dataset information
print(data.info())
# checking the missing value in each column
print(data.isnull().sum())
# distribution of legit transactions and fraud transaction
data['Class'].value_counts()
# 0 represents normal transaction & 1 represents fraud transaction
# seperating the data for analysis
legit = data[data.Class==0]
fraud = data[data.Class==1]
print(legit.shape)
print(fraud.shape)
# statistical measure of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())
# compare the values for both transaction
print(data.groupby('Class').mean())
# build sample dataset containg similar distribution  of both transaction
legit_sample = legit.sample(n=492)
new = pd.concat([legit_sample,fraud],axis=0)
print(new.head())
print(new['Class'].value_counts())
print(new.groupby('Class').mean())
# spliting the data into features and targets
x = new.drop(columns='Class',axis=1)
y = new['Class']
print(x)
print(y)
# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)
# Logistic regression
model = LogisticRegression
# training the regression model


# Assuming you have x_train and y_train prepared
# If not, load your data and split it into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LogisticRegression model
model = LogisticRegression()

# Training the regression model
model.fit(x_train, y_train)

# Make predictions on the training data
y_train_pred = model.predict(x_train)

# If you want to get predictions on a test set:
y_test_pred = model.predict(x_test)

# Calculate accuracy on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Calculate accuracy on the test data
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

