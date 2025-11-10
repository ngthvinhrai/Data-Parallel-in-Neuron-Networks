from NeuralNetworks.numpy_NeuralNetworks.Model import Sequential
from mpi4py import MPI
import subprocess
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class ParallelModel(Sequential):
    def __init__(self, Layers=[]):
        super().__init__(Layers)

    def backward(self, lr):
        dL_A = self.loss.deri
        wg = []
        bg = []
        for i in reversed(range(len(self.Layers))): 
            dL_A, weight_gradient, bias_gradient = self.Layers[i].backward(dL_A, self.optimizer[i], lr)
            wg.append(weight_gradient)
            bg.append(bias_gradient)
        return wg, bg

    def fit(self, X, Y, lr=0.1):
        self.forward(X)
        Y_hat = self.Layers[-1].getOutput()
        loss = self.loss(Y, Y_hat)
        weights_gradient, biases_gradient = self.backward(lr)
        weights_gradient.reverse()
        biases_gradient.reverse()

        return loss, weights_gradient, biases_gradient

if __name__ == "__main__":
    Y = np.load("MNIST_target.npy")
    print(Y)