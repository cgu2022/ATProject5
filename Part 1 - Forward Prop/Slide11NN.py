import numpy as np

class Model:

    def __init__(self):
         # Weights (Parameters)
         self.W1 = np.array(([[0.15, 0.3], [0.2, 0.35], [0.25, 0.4]]), dtype=float)
         self.W2 = np.array(([[0.5, 0.6, 0.7], [0.55, 0.65, 0.75]]), dtype=float)

        # Biases
         self.b1 = [[0.45], [0.45], [0.45]] # Length should be number of columns of X
         self.b2 = [[0.8], [0.8]]

    def sigmoid(self, z):
            return 1/(1+np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)
    
    def forward(self, X):
        # Propogate inputs through networks (i.e. Predicting with an input given current weight values)
        # print(self.W1)
        z2 = (np.dot(self.W1, X)) + self.b1 # Second layer Z
        print("z[1]:", z2)
        a2 = self.sigmoid(z2) # Second layer A
        print("a[1]:", a2)
        z3 = np.dot(self.W2, a2) + self.b2  # I discovered that numpy automatically "broadcasts" adding a scalar to a matrix
        print("z[2]:", z3)
        a3 = self.sigmoid(z3) # Third layer (Output layer) A
        print("a[2]:", a3)
        yHat = a3 # Calling it yHat for notation
        return yHat

    