import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

train_df = pd.read_csv("./predict-hourly-wage/income_training.csv")

train_x = train_df.drop(columns='compositeHourlyWages')

train_y = train_df['compositeHourlyWages']

model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(train_x.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile('adam', loss='mean_squared_error')

model.fit(x=train_x, y=train_y, validation_split=0.2, epochs=100)
