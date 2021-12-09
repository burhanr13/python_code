import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_cg


def create_Y(y, m, labels):
    Y = np.zeros([m, labels])
    for i in range(m):
        Y[i, int(y[i])] = 1
    return Y


def initialize_params(layer_sizes):
    T1 = np.random.rand(layer_sizes[0] + 1, layer_sizes[1]) - 0.5
    T2 = np.random.rand(layer_sizes[1] + 1, layer_sizes[2]) - 0.5
    T3 = np.random.rand(layer_sizes[2] + 1, layer_sizes[3]) - 0.5
    return roll_params(T1, T2, T3)


def roll_params(T1, T2, T3):
    return np.concatenate((T1, T2, T3), axis=None)


def unroll_params(params, layer_sizes):
    T1size = (layer_sizes[0]+1)*layer_sizes[1]
    T2size = (layer_sizes[1]+1)*layer_sizes[2]
    T3size = (layer_sizes[2]+1)*layer_sizes[3]
    [T1, T2, T3] = np.split(params, [T1size, T1size + T2size])
    T1 = T1.reshape(layer_sizes[0] + 1, layer_sizes[1])
    T2 = T2.reshape(layer_sizes[1] + 1, layer_sizes[2])
    T3 = T3.reshape(layer_sizes[2] + 1, layer_sizes[3])
    return T1, T2, T3


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(a):
    return a * (1 - a)


def forward_prop(X, m, T1, T2, T3):
    A0 = np.hstack((np.ones((m, 1)), X))

    A1 = sigmoid(A0 @ T1)
    A1 = np.hstack((np.ones((m, 1)), A1))

    A2 = sigmoid(A1 @ T2)
    A2 = np.hstack((np.ones((m, 1)), A2))

    A3 = sigmoid(A2 @ T3)

    return A1, A2, A3


def cost_function(params, X, Y, m, l, layer_sizes):
    T1, T2, T3 = unroll_params(params, layer_sizes)

    A1, A2, A3 = forward_prop(X, m, T1, T2, T3)

    T1reg, T2reg, T3reg = T1.copy(), T2.copy(), T3.copy()
    T1reg[:, 0], T2reg[:, 0], T3reg[:, 0] = 0, 0, 0

    cost = (-1 / m) * np.sum(Y * np.log(A3) + (1 - Y) * np.log(1 - A3))
    + (l / (2 * m)) * (np.sum(T1reg ** 2) +
                       np.sum(T2reg ** 2) + np.sum(T3reg ** 2))

    return cost


def calc_grads(params, X, Y, m, l, layer_sizes):
    T1, T2, T3 = unroll_params(params, layer_sizes)

    A1, A2, A3 = forward_prop(X, m, T1, T2, T3)

    A0 = np.hstack((np.ones((m, 1)), X))

    T1reg, T2reg, T3reg = T1.copy(), T2.copy(), T3.copy()
    T1reg[:, 0], T2reg[:, 0], T3reg[:, 0] = 0, 0, 0

    D3 = A3 - Y
    D2 = (D3 @ T3.T) * sigmoid_grad(A2)
    D2 = D2[:, 1:]
    D1 = (D2 @ T2.T) * sigmoid_grad(A1)
    D1 = D1[:, 1:]

    T1grad = (1 / m) * (D1.T @ A0).T + (l / m) * T1reg
    T2grad = (1 / m) * (D2.T @ A1).T + (l / m) * T2reg
    T3grad = (1 / m) * (D3.T @ A2).T + (l / m) * T3reg

    return roll_params(T1grad, T2grad, T3grad)


def grad_descent(params, X, Y, m, a, l, layer_sizes, iters, graph, save_params, filename=""):
    paramsc = params.copy()
    cost_func_vals = np.zeros(iters)
    for i in range(iters):
        cost = cost_function(paramsc, X, Y, m, l, layer_sizes)
        grads = calc_grads(paramsc, X, Y, m, l, layer_sizes)
        paramsc -= a * grads
        cost_func_vals[i] = cost
        print(".")

    if graph:
        plt.plot(np.arange(iters), cost_func_vals)
        plt.show()
    
    if(save_params):
        pd.DataFrame(paramsc).to_csv(filename, header=None, index=None)

    return paramsc, cost


def train_fmincg(params, X, Y, m, l, layer_sizes, save_params, filename=""):
    def callback(xk):
        print(cost_function(xk, X, Y, m, l, layer_sizes))

    res = fmin_cg(cost_function,
                  params, fprime=calc_grads,
                  args=(X, Y, m, l, layer_sizes), callback= callback)
    
    if(save_params):
        pd.DataFrame(res).to_csv(filename, header=None, index=None)

    return res


def calc_acc(X, y, m, params, layer_sizes):
    T1, T2, T3 = unroll_params(params, layer_sizes)
    A3 = forward_prop(X, m, T1, T2, T3)[2]
    pred = np.argmax(A3, 1)
    acc = np.mean((pred == y))
    return acc, pred


def find_a_l(X, Y, Xval, Yval, m, mval, params, layer_sizes, avec, lvec):
    count = 1
    costmat = np.zeros((avec.size, lvec.size))
    for i in range(avec.size):
        for j in range(lvec.size):
            print(str(count) + ": ", end="")
            a, l = avec[i], lvec[j]

            paramscalc, cost = grad_descent(
                params, X, Y, m, a, l, layer_sizes, 50, False)

            cost = cost_function(paramscalc, Xval, Yval, mval, l, layer_sizes)

            costmat[i, j] = cost

            count += 1

    aind, lind = np.unravel_index(np.argmin(costmat), costmat.shape)

    abest, lbest = avec[aind], lvec[lind]
    print(costmat)

    return abest, lbest


def testNN(X, m, imsize, pred):
    ex = np.random.randint(m)
    img = X[ex].reshape(imsize, imsize)
    print("Displaying image...")
    plt.imshow(img)
    plt.show()
    print(f"Neural network prediction: {pred[ex]}")


def test_new_img(imfile, paramfile, layer_sizes):
    img = plt.imread(imfile)
    img = np.mean(img, 2)
    plt.imshow(img)
    plt.show()
    img = img.flatten()
    img = img.reshape(1, img.size)
    params = pd.read_csv(paramfile, header=None).values
    T1, T2, T3 = unroll_params(params, layer_sizes)
    A1, A2, A3 = forward_prop(img, 1, T1, T2, T3)
    guess = np.argmax(A3)
    prob = np.max(A3)
    print(f"{guess}, {prob*100}%")
