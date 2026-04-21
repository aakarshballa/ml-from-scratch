import random
import math
class Value:
    def __init__(self, val=0, children=None):
        self.val = val 
        self.grad = 0
        self._backward = lambda:None
        self.children = children if children is not None else ()
    def __add__ (self, other):
        other = Value(other) if type(other) != Value else other
        out = Value(self.val + other.val, (self, other))
        def _backward():
            self.grad += out.grad 
            other.grad += out.grad 
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + other * -1
    
    def __mul__ (self, other):
        other = Value(other) if type(other) != Value else other
        out = Value(self.val * other.val, (self, other))
        def _backward():
            self.grad += out.grad * other.val
            other.grad += out.grad * self.val
        out._backward = _backward
        return out
    
    def __pow__(self, n):
        out = Value(self.val**n, (self, ))
        def _backward():
            self.grad += out.grad * n * (self.val ** (n-1))
        out._backward = _backward
        return out
    def tanh(self):
        out = Value(math.tanh(self.val), (self, ))
        def _backward():
            self.grad += (1 - out.val ** 2) * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return self * -1 + other 
    
    def __str__(self):
        return f"Value: {self.val}, Grad: {self.grad}"

    def backward(self):
        self.grad = 1.0
        visited = set()
        topo = []
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for v in reversed(topo):
            v._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.gauss(0,1)) for i in range(nin)]
        self.b = Value(0)
        self.nin = nin
    def parameters(self):
        return self.w + [self.b]
    def __call__(self, x):
        return (sum([w * i for w, i in zip(self.w, x)]) + self.b).tanh()

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for i in range(nout)]
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
class MLP:
    def __init__(self, nins):
        self.layers = [Layer(nin, nout) for nin, nout in zip(nins, nins[1:] + [1])]
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x[0]
    

        

