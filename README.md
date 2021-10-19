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
For data points with values <= 0.3 or >= 0.7, they are labelled as 0, else 1['outcome']. \

# Network Architecture
![alt text](https://github.com/kwquan/neural_network_from_scratch/main/nn.jpg)

