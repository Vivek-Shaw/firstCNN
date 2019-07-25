
#PART 1 Data preprocessing

#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing DataSet
dataset = pd.read_csv('Churn_Modelling.csv')

#creating the matrix of dataset
X = dataset.iloc[:, 3:13].values #x axis with the independent variable
Y = dataset.iloc[:,13].values    #y axis with the dependent variable

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#if some data in the data set are in string but we need them in integer we need to encode them 
#in integer with thw help of above library and the following code
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

#to categories the 2nd index i.e, not to make a priority
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#splitting the dataset into training and testing model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# scaling feature 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

# Part 2
#Making ANN 
#now get started by importing the keras libraries and some extensions
import keras
from keras.models import Sequential # to initialize ANN
from keras.layers import Dense #to create the multiple layers of ANN

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

#unit= the 1st hidden layers found by c=1/2(independent variable+dependent variable)
#kernel_initializer='uniform' gives the weight to the input nearer to 0 randomly 
#relu is the rectifier activation function
#input_dim is the number of inputs (independent variable)

#Adding second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#Compling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
#optimizer:it is used to get the best weight out the initialized weight at the start
#loss:it is used to optimize the losses that occurs while getting the output
#metrics:it is a criteria ti evalute to the model and it is used for improvization


#Fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size=10, epochs=100)
#batch_size:its the size of the dataset that are taken at a time for learning
#epochs: it is the number of times we make our nn to get the accuracy 

#Part-3 Making predictions and evaluating the model

#Predicting the set result
Y_pred = classifier.predict(X_test) #Predicting from the learned model
Y_pred=(Y_pred>0.5) #to get the result in true or false manner

#making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)















