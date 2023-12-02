import contextlib
import weakref
import numpy as np


class Config:
    enable_backprop = True


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0


    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'Variable(' + p + ')'


if __name__ == "__main__":
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x.shape)
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(x.shape)
    x = [1, 2, 3, 4]
    print(len(x))
    x = np.array([1, 2, 3, 4])
    print(len(x))
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(len(x))
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(len(x))