<h1>Data Parallel in Neuron Networks</h1><br>
Before running, clone this repository to Python site-package:<br>

```bash
git clone https://github.com/ngthvinhrai/NeuralNetworks.git
```

For example, my Python site-package directory:<br>

```bash
C:\Users\asus\AppData\Local\Programs\Python\Python312\Lib\site-packages
```

Then clone repository:<br>

```bash
https://github.com/ngthvinhrai/Data-Parallel-in-Neuron-Networks.git
```

Install MPI for python:<br>

```bash
pip install mpi4py
```

To run parallel program, use this command:<br>

```bash
mpiexec -n num_processors python main.py
```
