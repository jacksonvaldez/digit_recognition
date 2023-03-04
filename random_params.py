import numpy as np

# Creates a random set of weights
weights1 = np.random.uniform(-0.05, 0.05, (48, 784)) # Weights connecting the input layer and the hidden layer (48 x 784)
weights2 = np.random.uniform(-0.05, 0.05, (10, 48)) # Weights connecting the hidden layer and the output layer (10 x 48)

# Creates a set of biases, all 0 to start
biases1 = np.full(48, 0, dtype=np.float64).reshape(48, 1) # Biases connecting the input layer and the hidden layer (48 x 1)
biases2 = np.full(10, 0, dtype=np.float64).reshape(10, 1) # Biases connecting the hidden layer and the output layer (10 x 1)

epoch = [0]

np.save('trained_params/weights1.npy', weights1)
np.save('trained_params/weights2.npy', weights2)
np.save('trained_params/biases1.npy', biases1)
np.save('trained_params/biases2.npy', biases2)
np.save('epoch.npy', epoch)