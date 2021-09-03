import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv("./affairsdata/data.csv").to_numpy()

X = data[:,1:-1]

Y = data[:,-1]

scaler = preprocessing.MinMaxScaler()

data = scaler.fit_transform(X, y=Y)

gilbert = pd.read_csv("./affairsdata/gilbert.csv")

gilbert = scaler.transform(gilbert)

X, Xtest, Y, Ytest = train_test_split(X, Y, test_size=0.3, random_state=101)

model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile('adam', loss='mean_squared_error')

model.fit(x=X, y=Y, validation_split=0.3, epochs=100)

print(model.evaluate(x=Xtest,y=Ytest))

print(model.predict(gilbert))