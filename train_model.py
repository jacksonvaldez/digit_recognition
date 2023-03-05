from neural_network import NeuralNetwork
from mnist_data.load import load_mnist
import numpy as np
import pdb

def get_learn_rate(epoch):
	# if epoch > 40:
	# 	epoch = 40

	lr = 0.1 * (0.1 ** (epoch / 10))

	return lr


images_test, labels_test = load_mnist('mnist_data', kind='t10k')
images_train, labels_train = load_mnist('mnist_data', kind='train')

for x in range(500):
	print('------------------- Model Training -------------------')
	epoch = np.load('epoch.npy')[0]
	learn_rate = get_learn_rate(epoch)
	print('Epoch:', epoch)
	print('Learn Rate:', learn_rate)

	weights1 = np.load('trained_params/weights1.npy')
	weights2 = np.load('trained_params/weights2.npy')
	biases1 = np.load('trained_params/biases1.npy')
	biases2 = np.load('trained_params/biases2.npy')

	neural_net = NeuralNetwork(weights1, weights2, biases1, biases2)
	neural_net.train(images_train, labels_train, learn_rate)

	epoch += 1
	np.save('epoch.npy', [epoch])

	np.save('trained_params/weights1.npy', neural_net.weights1)
	np.save('trained_params/weights2.npy', neural_net.weights2)
	np.save('trained_params/biases1.npy', neural_net.biases1)
	np.save('trained_params/biases2.npy', neural_net.biases2)

	print('Model Trained!')
	print(f"Testing Accuracy: {neural_net.accuracy(images_test, labels_test)}%")
