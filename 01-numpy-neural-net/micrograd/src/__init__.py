from layers import *
from network import *
from optimizers import *
from train import *
from torchvision import datasets, transforms
import numpy as np

def main():
    a = Value(5)
    b = Value(6)
    d = a + 5
    c = a * b
    e = c + d
    
    e.backward()

    xs = [[-1.0, 2.0, 3.0, 5.0],
          [4.0, 2.0, 1.0, -1.0],]
    # need to be between -1 and 1 because of tanh
    ys = [1.0, -1.0]

    m = MLP([4, 8, 8, 4])

    # ===========================================
    # TRAINING LOOP
    for i in range(5000):
        
        preds = [m(x) for x in xs]

        loss = sum([(pred - y)**2 for pred, y in zip(preds, ys)])

        # zero grad 
        for p in m.parameters():
            p.grad = 0

        # do backward function
        loss.backward()

        # increment s
        for p in m.parameters():
            p.val += -p.grad * 0.01        
        if (i % 100 == 0):
            print(f"Epoch: {i}, Loss: {loss.val}")
    # =======================================



if __name__ == "__main__":
    main()