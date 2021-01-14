import numpy as np
from Slide11NN import Model

#X = np.array(([[0.05, 0.05, 0.05], [0.1, 0.1, 0.1]]), dtype=float) # 2, 3
X = np.array(([[0.05, 0.05, 0.05], [0.1, 0.1, 0.1]]), dtype=float) # 2, 3

model = Model()

output = model.forward(X)