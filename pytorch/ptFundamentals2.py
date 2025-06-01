import torch
import numpy as np

"""INTRODUCTION PART 2"""

### MANIPULATING TENSORS ###
#1. Addition
#2. Subtraction
#3. Multiplication / Matrix multiplication
#4. Division
tensor = torch.tensor([1,2,3])
tensor + 10
tensor.add(10)


#Matrix multiplication (dot product)
torch.matmul(tensor, tensor) #Skalarprodukt wegen zwei Vektoren

#1. The inner dimensions must match
# (3,2) @ (2,3) perfect
# (3,2) @ (3,2) NOO!!!

#2. The resulting matrix has the shape of outer dimensions
# (3,2) @ (2,3) = (3,3)
# (2,3) @ (3,2) = (2,2) 
torch.matmul(torch.rand(10,3), torch.rand(3,10))

#One of the most common errors in deep learning: shape errors
tensorA = torch.tensor([[1,2],
                        [3,4],
                        [5,6]])
tensorB = torch.tensor([[7,8],
                        [9,10],
                        [11,12]])
#To fix error, we use torch.transpose or .T (we switch Zeilen und Spalten)
torch.matmul(tensorA, tensorB.T)


## RESHAPING, STACKING, SQUEEZING, UNSQUEEZING
# reshaping - reshapes an input tensor to a defined shape
# view - return a view of an input tensor but keep the same memory
# stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# squeezing - removes all one-dimensions from a tensor
# unsqueeze - add an one-dimesion to a target tensor
# permute - return a view of the input with dimensions permuted (swapped) in a certain way

x = torch.arange(0, 100, 10)
tensorX = torch.arange(1.,10.)
tensorXreshaped = torch.reshape(tensorX, shape=(1,9))
tensorZ = x.view(10,1) #Changing anything from z, x will change too

tensorXstacked = torch.stack([x,x], dim=0) #Dim = 0 is like axis=0 (vertical) or dim=1 (horizontal)

tensorXsqueezed = torch.squeeze(tensorXreshaped)
tensorXunsqueezed = torch.unsqueeze(tensorXsqueezed, dim=0)

xOriginal = torch.rand(size=(224,224,3))
xPermuted = torch.permute(xOriginal, dims=(2,0,1)) #Rearranges the dimensions


##INDEXING (selecting data from tensors)
x = torch.arange(1,10).reshape(1,3,3)
x[0]       #First dimension
x[0, 0]    #First dimension, first line
x[0,0,0]   #First dimension, first line, first value
x[:,0]     #All dimension, first line
x[:,:,1]   #Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension


#PYTORCH TENSORS & NUMPY
#Data in NumPy, want in PyTorch Tensor -> 'torch.from_numpy(ndarray)'
#Pytorch into Numpy -> torch.Tensor.numpy()

array = np.arange(1., 8.)
tensor = torch.from_numpy(array).type(torch.float32)

tensor2 = torch.ones(7)
numpyTensor = tensor2.numpy()

## REPRODUCIBILITY ###
torch.manual_seed(42)

#Running tensors and PyTorch objects on the GPUs
# 1. Google Colab - free GPU
# 2. Kaggle - free GPU
# 3. Use your own NVIDIA GPU

#Set up device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"
tensorE = torch.tensor([1,2,3])
tensorGPU = tensor.to(device) #Switching to GPU 

#Moving tensors back to CPU
tensorCPU = tensorGPU.cpu()
tensorGPU.numpy()


#EXERCISES
torch.manual_seed(7)
randomTensor = torch.rand(1,1,1,10)
randomTensor2 = torch.squeeze(randomTensor)
print(randomTensor)
print(randomTensor.shape)
print(randomTensor2)
print(randomTensor2.shape)
