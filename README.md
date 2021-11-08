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
We use a simple network architecture comprising of 1 input layer, 1 hidden layer[2 nodes] & 1 output layer[learning rate = 0.005, steps = 5000]: 
![alt text](https://github.com/kwquan/neural_network_from_scratch/blob/main/nn.jpg)

# Forward Propagation 
Hidden layer consist of 2 nodes. \
First node has weight w1 & bias b1. Input is multipled to the weight & added to the bias according to: \
x1 = input * w1 + b1 \
Second node has weight w2 & bias b2. Input is multipled to the weight & added to the bias according to:\
x2 = input * w2 + b2 

Next, both x1 & x2 are passed to the respective activation functions: \
y1 = max(0,x1) \
y2 = max(0,x2) \
Note that we use Relu for both nodes. 

Finally, we multiply y1 & y2 to the respective weights & sum them according to: \
top = y1 * w3 \
bottom = y2 * w4 \
prediction = max(0,top + bottom + b3) 

# Backward Propagation
First, we define error as : \
SSR = sum(observed - (top + bottom + b3))^2 = sum(observed - (w3y1 + w4y2 +b3))^2 \

Let's look at b3: \
dSSR/db3 = -2 * sum(observed - (w3y1 + w4y2 +b3)) \
b3 -= learning rate * dSSR/db3 

Let's look at w3 & w4: \
dSSR/dw3 = -2 * sum(observed - (w3y1 + w4y2 +b3)) * y1 \
dSSR/dw4 = -2 * sum(observed - (w3y1 + w4y2 +b3)) * y2 \
We update w3 & w4 as follows: \
w3 -= learning rate * dSSR/dw3 \
w4 -= learning rate * dSSR/dw4 

Let's look at b1 & b2: \
Since we are using Relu activation functions, the derivatives will be 1 most of the time[when y1 = x1] & 0 rarely[if max(0,x1) = 0] \
dSSR/db1[Using chain rule] = dSSR/dPredicted * dPredicted/dy1 * dy1/dx1 * dx1/db1 \
                           = -2 * sum(observed - predicted) * w3 * (1 or 0) \
dSSR/db2[Using chain rule] = dSSR/dPredicted * dPredicted/dy2 * dy2/dx2 * dx2/db2 \
                           = -2 * sum(observed - predicted) * w4 * (1 or 0) \
We update b1 & b2 as follows: \
b1 -= learning rate * dSSR/db1 \
b2 -= learning rate * dSSR/db2 \     

Let's look at w1 & w2: \
dSSR/dw1[Using chain rule] = dSSR/dPredicted * dPredicted/dy1 * dy1/dx1 * dx1/dw1 \
                           = -2 * sum(observed - predicted) * w3 * (1 or 0) * input \
dSSR/dw2[Using chain rule] = dSSR/dPredicted * dPredicted/dy2 * dy2/dx2 * dx2/dw2 \
                           = -2 * sum(observed - predicted) * w4 * (1 or 0) * input \
We update w1 & w2 as follows: \
w1 -= learning rate * dSSR/dw1 \
w2 -= learning rate * dSSR/dw2 \ 

Phew! That was a lot of math. If u made it this far, you should be proud of yourself! 

# Model Performance
Plotting the SSR against steps, one can see that the model performance plateaus very quickly: \
![alt text](https://github.com/kwquan/neural_network_from_scratch/blob/main/error.png)

Of course, this simple model is insufficient to predict the classes well. \
Nonetheless, it's a good start in getting to understand how neural networks work!
