import numpy as np


class Variable:
    def __init__(self, x):
        self.data = x
    

class Function:
    def __call__(self, input_):
        x = input_.data
        y = self.forward(x)
        output_ = Variable(y)
        return output_
    
    def forward(self, x):
        raise NotImplementedError
    

class Square(Function):
    def forward(self, x):
        return x ** 2
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)