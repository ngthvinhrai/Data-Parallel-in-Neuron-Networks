from sklearn.datasets import fetch_openml
import numpy as np

# Tải MNIST từ OpenML
mnist = fetch_openml('mnist_784', version=1)

# Dữ liệu ảnh và nhãn
X, y = mnist['data'], mnist['target']

# Chuyển nhãn về kiểu số nguyên
y = y.astype(int)

np.save("MNIST_data.npy", X)
np.save("MNIST_target.npy", y)
