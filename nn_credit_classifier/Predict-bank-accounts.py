# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('.../Bank_Predictions.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# ------ Part-1: Data preprocessing ----------

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# ------- Part-2: Build the ANN --------

# import keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# You might use the following parameters: activation: 'relu' and/or 'sigmoid', optimization function is 'adam', loss function is 'binary_crossentropy', number of epochs is 100, samples per epoch are 10) 

# Adding the input layer and the first hidden layer
# The missing line of code here....

# Adding second hidden layer
# The missing line of code here....

# Adding output layer
# The missing line of code here....

# Compiling the ANN
# The missing line of code here....

# Fitting the ANN to the training set
# The missing line of code here....

# Predicting the Test set results
# The missing line of code here....

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

