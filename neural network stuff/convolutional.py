from keras.models import Sequential
from keras import layers

import pandas as pd
import numpy as np

model = Sequential()

model.add(layers.Conv2D(32, (3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

data = pd.read_csv("./mnist data/mnist_train.csv", header=None).to_numpy()
X = data[:, 1:]
y = data[:, 0]
test_data = pd.read_csv("./mnist data/mnist_test.csv", header=None).to_numpy()
Xtest = test_data[:, 1:]
ytest = test_data[:, 0]

X = X.reshape(60000,28,28,1)
Xtest = Xtest.reshape(10000,28,28,1)

X = X.astype('float32')/255
Xtest = Xtest.astype('float32')/255

def create_Y(y, m, labels):
    Y = np.zeros([m, labels])
    for i in range(m):
        Y[i, int(y[i])] = 1
    return Y

Y = create_Y(y,60000,10)
Ytest = create_Y(ytest,10000,10)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X,Y,epochs=5,batch_size=64)

print(model.evaluate(x=Xtest,y=Ytest))
