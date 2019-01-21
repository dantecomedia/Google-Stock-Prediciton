#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the training set
dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1:2]

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

#create a datastructure with 60 timesteps and 1 output
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
    
#Reshaping
X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

#Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
 
#Intialzing the RNN
regressor=Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer="adam",loss="mean_squared_error")
regressor.fit(X_train,y_train, epochs=100,batch_size=32)
