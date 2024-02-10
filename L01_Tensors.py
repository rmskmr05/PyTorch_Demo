import torch
import numpy as np

### Tensor Manipulation
x = torch.zeros(4,3)
print("Tensdor x:\n", x)
print("Data Type of the tensor x:", x.dtype)
print("Size of tensor x:", x.shape)

### Defined a tensor with specific data type
x = torch.ones((4, 5), dtype=torch.int16) #with 4 rows and 5 columns
print("\nTensdor x:\n", x)
print("Data Type of the tensor x:", x.dtype)

### Generate a random number
r1 = torch.rand(10)
print("Random Number r1:", r1)
r2 = torch.rand(10)
print("Random Number r2:", r2)

### Generate random number with (with same seed generates same random number)
torch.manual_seed(100)
r1 = torch.rand((4, 3))
print("\nRandom Number r1:\n", r1)
r2 = torch.rand((4, 3))
print("Random Number r2 (different from r1):\n", r2)
torch.manual_seed(100)
r3 = torch.rand((4, 3))
print("Random Number r3 (same as r1):\n", r3)

### Tensor Arithmatic
r1 = torch.rand(5, 5)
print("\nr1:\n", r1)

ones = torch.ones(4, 5)
print(ones)
twos = torch.ones(4, 5) * 2
print(twos)
threes = ones + twos
print(threes)
## Matrics Multiplication
dm = torch.matmul(twos, r1)
print("Dot Multiplication of r1 of size {} and twos of size {}:\n".format(r1.shape, twos.shape), dm)

r = 2 * r1 - 1  # value range [-1. 1]
print("\nr:\n", r)
print("Absolute of r:\n", abs(r))
print("Inverse of sin(r) = (asin(r)) in radian:\n", torch.asin(r))
print("Inverse of cin(r) = (acin(r)) in radian:\n", torch.acos(r))

## Linear Algebra Operation
print("\nDeterminant of r:", torch.det(r))
print("SVD of r:\n", torch.svd(r))

## statistical operations
print("\nAverage and Standard deviation of r:", torch.std_mean(r))
print("Max of r = {}, Min of r = {}".format(torch.max(r), torch.min(r)))

### Convert a tensor to numpy array
a_r1 = r.numpy()
print("\nNumpy array of r using torch:\n", a_r1, type(a_r1))
a_r2 = np.array(r1)
print("\nNumpy array of r using numpy:\n", a_r2, type(a_r2))

## Conver a numpy array to tensor
t_r = torch.tensor(a_r1)
print("\ntensor from numpy array a_r:\n", t_r, type(t_r))