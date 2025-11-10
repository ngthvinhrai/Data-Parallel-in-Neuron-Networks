from model import ParallelModel
from NeuralNetworks.numpy_NeuralNetworks.Layer import Dense
from NeuralNetworks.numpy_NeuralNetworks.Model import Sequential
from NeuralNetworks.numpy_NeuralNetworks.Activation import Relu, Softmax
from NeuralNetworks.numpy_NeuralNetworks.Loss import CrossEntropy, SpareCrossEntropy
from NeuralNetworks.numpy_NeuralNetworks.Optimizer import GradientDescent, Momentum
import numpy as np
import json
import time

np.random.seed(18)
def main():
    x = np.load("MNIST_data.npy")
    x = x/255
    x_train = x[:42000]
    x_val, x_test = x[42000:56000], x[56000:]
    y = np.load("MNIST_target.npy")
    y_train = y[:42000]
    y_val, y_test = y[42000:56000], y[56000:]
    model = Sequential([
        Dense(input_shape=784, output_shape=16, activation=Relu()),
        Dense(input_shape=16, output_shape=16, activation=Relu()),
        Dense(input_shape=16, output_shape=10, activation=Softmax()),
    ])
    model.compile(loss=SpareCrossEntropy(), optimizer=Momentum(norm=True))
    t1 = time.time()
    his = model.fit(x_train,y_train, val_data=[x_val, y_val], batch_size=1024, epochs=25, lr=0.01)
    print(f"Time: {time.time() - t1:.4f}")
    model.save_weights('sequential_model')

    with open('sequential_his.json', 'w', encoding='utf-8') as f:
        json.dump(his, f, indent=2)

if __name__ == "__main__":
    main()