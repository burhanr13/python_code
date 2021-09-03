import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simple_nn as nn

m = 10000
mval = 0
mtest = 10000

imsize = 28
n = imsize ** 2
labels = 10

layer_sizes = np.array([n, 50, 50, labels])

grad_descent_iters = 500
avec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
lvec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])

l = 0.3

print("Loading and processing data...")

data = pd.read_csv("./mnist data/mnist_train.csv", header=None).to_numpy()

X = data[0:m, 1:]
y = data[0:m, 0]
X = X / 255
Y = nn.create_Y(y, m, labels)

Xval = data[m:m+mval, 1:]
yval = data[m:m+mval, 0]
Xval = Xval / 255
Yval = nn.create_Y(yval, mval, labels)

print("Displaying examples of data...")

fig = plt.figure()
for i in range(1,31):
    img = X[np.random.randint(0,m)].reshape(imsize, imsize)
    fig.add_subplot(5, 6, i)
    plt.imshow(img)
plt.show()

print("Initializing neural network parameters...")

nn_params = nn.initialize_params(layer_sizes)

"""print("Finding best alpha and lambda values for neural network...")

abest, lbest = nn.find_a_l(X, Y, Xval, Yval, m, mval,
                          nn_params, layer_sizes, avec, lvec)

print(f"Best alpha: {abest}\nBest lambda: {lbest}")
print("Training neural network with best alpha and lambda values and gradient descent...")

nn_params, cost = nn.grad_descent(
   nn_params, X, Y, m, 1, 0, layer_sizes, grad_descent_iters, True, True, "./trained_params.csv")

print(f"Neural network trained\nFinal cost: {cost}")"""

print("Training neural network with fmin_cg...")

nn_params = nn.train_fmincg(nn_params, X, Y, m, l, layer_sizes, True, "./trained_params.csv")

print("Neural network trained.")

print("Calculating training and validation accuracy...")

acc, pred = nn.calc_acc(X, y, m, nn_params, layer_sizes)
if(mval != 0):
    valacc, valpred = nn.calc_acc(Xval, yval, mval, nn_params, layer_sizes)

print(f"Training accuracy: {acc*100}%")

# print(f"Validation accuracy: {valacc*100}%")

print("Loading and processing test data...")

test_data = pd.read_csv("./mnist data/mnist_test.csv", header=None).to_numpy()

Xtest = test_data[0:mtest, 1:]
ytest = test_data[0:mtest, 0]
Xtest = Xtest / 255
Ytest = nn.create_Y(ytest, mtest, labels)

print("Calculating test accuracy...")

testacc, testpred = nn.calc_acc(Xtest, ytest, mtest, nn_params, layer_sizes)

print(f"Test accuracy: {testacc*100}%")

print("Testing neural network...")

ans = ""

while ans != "q":
    nn.testNN(Xtest, mtest, imsize, testpred)
    ans = input("Type 'q' to quit, otherwise continue: ")
