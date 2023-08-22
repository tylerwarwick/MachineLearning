import pandas as pd
import numpy as np


#Import dataset from MNIST
trainingData = pd.read_csv('../ML/README.md')
testingData = pd.read_csv('./MNIST_CSV/mnist_test.csv')

#Transform into numpy array
trainingData = np.array(trainingData)
testingData = np.array(testingData)

#Store dimensions so that we can pull expected values out
m, n = trainingData.shape
j, k = testingData.shape 

### TRAINING DATA ###
#Transpose matrix to have each image be a column vector
trainingData = trainingData.T
expectedOutput_Train = trainingData[0]
data_Train = trainingData[1:n]
data_Train = data_Train / 255.


### TESTING DATA ###
testingData = testingData.T
print(testingData.shape)
expectedOutput_Test = testingData[0]
print(expectedOutput_Test.shape):
data_Test = testingData[1:k]
data_Test =  data_Test / 255


#Define inital weights and biases with values between -0.5 and 0.5
def init_params():
    #One image makes 784 greyscale values
    #The dot product then reduces those down to 10 nodes in the hidden layer
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
   
    return W1, b1, W2, b2


### As an aside we use activation functions so that the nodes ###
### are not just linear combinations ###

#Our grayscale points will all be less than 0 
#We can just return the value if greater or equal to 0.5
#If it's a negative number return 0
def ReLU(Z):
    return np.maximum(Z, 0)

#Need to return nodes with decimal values between 0 and 1
#These are our confidence in the predictions
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
#Apply weights n biases from input nodes to output nodes
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1) 
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#Derivative of ReLu is 0 or 1 because it's a linear function
def ReLU_deriv(Z):
    #True equals a value of 1 and false to 0
    return Z > 0


#Need to encode our expectations as 10 element vector
#9 nodes should be 0 except for the actual digit
def expectedOutputEncoded(Y):
    eoy = np.zeros((Y.size, Y.max() + 1))
    eoy[np.arange(Y.size), Y] = 1
    eoy = eoy.T
    return eoy


#Define cost function and bind gradient descent that hopefully converges to global minimum
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    eoy = expectedOutputEncoded(Y)
    dZ2 = A2 - eoy
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2

#Update weights n biases with gradient descent vector times the learning rate
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate):
    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1    
    W2 = W2 - learningRate * dW2  
    b2 = b2 - learningRate * db2    
    return W1, b1, W2, b2

#Output node with largest value is networks prediction
def get_predictions(A2):
    return np.argmax(A2, 0)

#Compare expectedOutput and predictions vectors for accuracy
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

#Loop through forward and backward propogation in hopes of convergence
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(data_Train, expectedOutput_Train, 0.10, 500)

##Test weights n biases with test data
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, data_Test)
predictions = get_predictions(A2)
print(get_accuracy(predictions, expectedOutput_Test))

