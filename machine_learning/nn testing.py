import numpy as np 
import matplotlib.pyplot as plt
import simple_nn as nn
import pandas as pd 

layer_sizes = np.array([784, 50, 50, 10])

nn.test_new_img("./testimg.png", "./trained_params.csv", layer_sizes)

