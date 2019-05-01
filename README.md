# Project Title:

##  To build a handwritten digit classifier using Neural Network.

### In this problem, we will use our algorithm to train a model on the MNIST dataset. We will restrict our attention to only two classes: digits 6 and 8. 

* We will Implemented a generic neural network architecture. 
* * Implemnt the backpropagation algorithm to train the network. 
* * Will train the network using Stochastic Gradient Descent (SGD) where the batch size is 100 in SGD.  
* * Will Correctly implemented gradient descent by making use of sub-gradient at z = 0.
* We will create a network with a single hidden layer having 100 units. We will vary the learning rate(n) being inversely proportional to number of iterations(t) i.e where t is the current learning iteration number. Choose an appropriate stopping criterion based
on the change in value of the error function.
* * We will Use sigmoid as the activation function in our network.
* * Next we will use ReLU as the activtion instead of the sigmoid function. ReLU is defined using the function: g(z) = max(0; z).we will still use sigmoid in the final (output) layer since we are dealing with Boolean output.
* * Will Plot the decision boundary along with the test samples.
* * Will Report and plot the accuracy on the training and the test sets, time taken to train the network. 
