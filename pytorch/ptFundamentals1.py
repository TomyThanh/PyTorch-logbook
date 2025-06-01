import torch

"""INTRODUCTION TO TENSORS"""
##Creating Tensors
#1. Scalar
scalar = torch.tensor(7)
scalar.ndim 
scalar.item() #get tensor back as Python int

#2. Vector
vector = torch.tensor([7,7])
vector.ndim
vector.shape

#3. MATRIX
MATRIX = torch.tensor([[7,8], [9,10]])
MATRIX.ndim
MATRIX.shape

#4. Tensor
TENSOR = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]]])
TENSOR.ndim
TENSOR.shape

##RANDOM TENSORS
#Create a random tensor of size(3,4) / shape(3,4)
randomTensor = torch.rand(3,4)
randomTensor.ndim

#Create a random tensor with similar shape to an image tensor
randomImageSizeTensor = torch.rand(size=(3,224,224)) #height, width, color channel

#Zeros and Ones
zeros = torch.zeros(size=(3,4))
ones = torch.ones(size=(3,4))
ones.dtype

##RANGE OF TENSORS
torch.arange(0,11) #from 0 to 10

##TENSOR DATATYPES
float32Tensor = torch.tensor([3.0,6.0,9.0], dtype=torch.float16, #Changing dtype
                             device=None,  #What device is your tensor on
                             requires_grad=False) #Whether or not to track gradients with this tensors operations

float16Tensor = float32Tensor.type(torch.float16) #Changing float32Tensor into float16

#Getting informations from tensors
#1. Tensor datatype - to get dtype you can use 'tensor.dtype'
#2. Tensor shape - to get shape you ucan use 'tensor.shape'
#3. Tensor device - to get device from a tensor, you can use 'tensor.device'

#Find out details about some tensor
someTensor = torch.rand(3,7)
someTensor.dtype
someTensor.shape
someTensor.device #CPU or GPU / TPU

#Finding the min, max, mean, sum, etc. (tensor aggregation)
x = torch.arange(0, 100, 10)
x.min(), torch.min(x) 
x.max(), torch.max(x)
torch.mean(x.type(dtype=torch.float32)), x.type(torch.float32).mean() #To find out the mean we need float
torch.sum(x), x.sum()
x.argmax() #Find the index that has max value
x.argmin() #Find the index that has min value

