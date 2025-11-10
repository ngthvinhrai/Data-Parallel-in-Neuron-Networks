from model import ParallelModel
from NeuralNetworks.numpy_NeuralNetworks.Layer import Dense
from NeuralNetworks.numpy_NeuralNetworks.Model import Sequential
from NeuralNetworks.numpy_NeuralNetworks.Activation import Relu, Softmax
from NeuralNetworks.numpy_NeuralNetworks.Loss import SpareCrossEntropy
from NeuralNetworks.numpy_NeuralNetworks.Optimizer import Momentum
import numpy as np
from mpi4py import MPI  
import time
import json

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(18)
def main():
    start_time = MPI.Wtime()
    EPOCHS = 100
    lr = 0.01

    if rank == 0:
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        MODEL_PATH = "parallel_model_6_100_001"
        x = np.load("MNIST_data.npy")
        x = x/255
        x_train = x[:42000]
        x_val, x_test = x[42000:56000], x[56000:]
        y = np.load("MNIST_target.npy")
        y_train = y[:42000]
        y_val, y_test = y[42000:56000], y[56000:]

        model = ParallelModel([
            Dense(input_shape=784, output_shape=16, activation=Relu()),
            Dense(input_shape=16, output_shape=16, activation=Relu()),
            Dense(input_shape=16, output_shape=10, activation=Softmax()),
        ])
        model.compile(loss=SpareCrossEntropy(), optimizer=Momentum(norm=True))
    else:
        model = None
    
    model = comm.bcast(model, root=0)

    for epoch in range(EPOCHS):
        if rank == 0:
            indices = np.arange(42000)
            np.random.shuffle(indices)
            x = x_train[indices]
            y = y_train[indices]
            chunks_x = np.array_split(x, size, axis=0)
            chunks_y = np.array_split(y, size, axis=0)
            for i in range(1, size):
                comm.send([chunks_x[i], chunks_y[i]], dest=i)
            X, Y = chunks_x[0], chunks_y[0]
        else:
            X, Y = comm.recv(source=0)

        loss, weights_gradient, biases_gradient = model.fit(X,Y,lr=lr)
        gathered_weights_gradient = comm.gather(weights_gradient, root=0)
        gathered_biases_gradient = comm.gather(biases_gradient, root=0)
        comm.Barrier()

        if rank==0:
            for i in range(len(model.Layers)):
                layer_weight_gradient = [grad_w[i] for grad_w in gathered_weights_gradient]
                layer_bias_gradient = [grad_b[i] for grad_b in gathered_biases_gradient]
                grad_W = np.sum(layer_weight_gradient, axis=0)/size
                grad_b = np.sum(layer_bias_gradient, axis=0)/size
                model.Layers[i].W -= lr*grad_W
                model.Layers[i].b -= lr*grad_b

            y_hat = model.predict(x)
            loss = model.loss(y, y_hat)
            count = np.bincount(np.argmax(y_hat, axis=1) == y)
            accuracy = count[1]/len(y)

            y_val_hat = model.predict(x_val)
            val_loss = model.loss(y_val, y_val_hat)
            val_count = np.bincount(np.argmax(y_val_hat, axis=1) == y_val)
            val_accuracy = val_count[1]/len(y_hat)

            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            print(f"Epoch {epoch+1}/{EPOCHS}: loss = {loss:.4f} - accuracy = {accuracy:.4f} - val_loss = {val_loss:.4f} - val_accuracy = {val_accuracy:.4f}", flush=True)

        else:
            model = None
            
        model = comm.bcast(model, root=0)
        comm.Barrier()

    comm.Barrier() 
    end_time = MPI.Wtime()
    
    if rank == 0:
        total_runtime = end_time - start_time
        print("-" * 30)
        print(f"Total Program Runtime: {total_runtime:.4f} seconds")
        print("-" * 30)
        model.save_weights(MODEL_PATH)
        
        with open('parallel_his.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    return

def main2():
    start_time = time.time()
    x = np.load("MNIST_data.npy")
    x = x[:42000]
    x = x/255
    y = np.load("MNIST_target.npy")
    y = y[:42000]
    model = Sequential([
        Dense(input_shape=784, output_shape=16, activation=Relu()),
        Dense(input_shape=16, output_shape=16, activation=Relu()),
        Dense(input_shape=16, output_shape=10, activation=Softmax()),
    ])
    model.compile(loss=SpareCrossEntropy(), optimizer=Momentum(norm=True))
    model.fit(x,y, batch_size=1000, epochs=20, lr=0.01)
    end_time = time.time()
    total_runtime = end_time - start_time
    print("-" * 30)
    print(f"Total Program Runtime: {total_runtime:.4f} seconds")
    print("-" * 30)

if __name__ == '__main__':
    main()