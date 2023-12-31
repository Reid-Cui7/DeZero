import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

if __name__ == "__main__":
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
    x.data = np.array(2.0)
    print(x.data)
########################
    x = np.array(1)
    print(x.ndim)
    x = np.array([1, 2, 3])
    print(x.ndim)
    x = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(x.ndim)