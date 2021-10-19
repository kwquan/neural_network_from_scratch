# Neural Network from Scratch
In this notebook, we shall create a neural network from scratch using only Pandas & NumPy. \
This is to enable us to better understand the workings behind forward & backward propagations in a neural network 
instead of doing it blindly using some deep learning package. 

# Objective
This is a binary classification problem. \
Given 100 data points, the model aims to predict a class for every data point[0 or 1] 

# Input Data
We generate 100 random data points in the range 0-1. \
Next, we create a dataframe and name the data points 'dosage'. \
For data points with values <= 0.3 or >= 0.7, they are labelled as 0, else 1['outcome']. 

# Network Architecture
We use a simple network architecture comprising of 1 input layer, 1 hidden layer[2 nodes] & 1 output layer. 
![alt text](https://github.com/kwquan/neural_network_from_scratch/blob/main/nn.jpg)

# Forward Propagation 
Hidden layer consist of 2 nodes. \
First node has weight w1 & bias b1. Input is multipled to the weight & added to the bias according to: \
x1 = input*w1 + b1 \
Second node has weight w2 & bias b2. Input is multipled to the weight & added to the bias according to:\
x2 = input*w2 + b2 \

Next, both x1 & x2 are passed to the respective activation functions: \
y1 = max(0,x1) \
y2 = max(0,x2) \
Note that we use Relu for both nodes. \

Finally, we multiply y1 & y2 to the respective weights & sum them according to: \
top = y1*w3 \
bottom = y2*w4 \
output = top + bottom + b3 


