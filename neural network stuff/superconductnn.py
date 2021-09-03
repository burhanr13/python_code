import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv("./superconduct.csv").to_numpy()

X = data[:, :-1]
Y = data[:, -1]

scaler = preprocessing.MinMaxScaler()

X = scaler.fit_transform(X, y=Y)

X, Xtest, Y, Ytest = train_test_split(X, Y, test_size=0.25)

model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile('adam', loss='mean_squared_error')

model.fit(x=X, y=Y, epochs=1000, validation_split=0.2)

print(model.evaluate(x=Xtest,y=Ytest))
