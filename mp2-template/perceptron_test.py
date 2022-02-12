import numpy as np

from perceptron import trainPerceptron

a = np.array([1,1,1,1])
a = a * 5
print(a)

train_set = np.array([[0,2], [-2,1], [3,0]])
train_label = np.array([1, -1, 1])

# print(trainPerceptron(train_set, train_label, 1, 1))
