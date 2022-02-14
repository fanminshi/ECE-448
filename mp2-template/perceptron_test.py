import numpy as np

from perceptron import trainPerceptron

import torch.nn as nn


# a = np.array([1,1,1,1])
# a = a * 5
# print(a)

# train_set = np.array([[0,2], [-2,1], [3,0]])
# train_label = np.array([1, -1, 1])

import torch
a = torch.tensor([[1.0,1.0],[2.0, 2.0]])
print(a)
std, mean = torch.std_mean(a)
print(std, mean)
a = (a - mean) / std
a = a / std
print(a)

model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
print(model(a))
# print(trainPerceptron(train_set, train_label, 1, 1))
