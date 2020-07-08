# Project Title:

##  To build a handwritten digit classifier using Neural Network.

### In this problem, we will use our algorithm to train a model on the MNIST dataset. We will restrict our attention to only two classes: digits 6 and 8. 

* We will Implemented a generic neural network architecture. 
* * Implemnt the backpropagation algorithm to train the network. 
* * Will train the network using Stochastic Gradient Descent (SGD) where the batch size is 100 in SGD.  
* * Will Correctly implemented gradient descent by making use of sub-gradient at z = 0.

* Visualizing Decision Boundary: 
* *  Will Train the network using the logistic regression learner available from the scikit-learn library of Python: logistic regression learner. Will Compute the train and test set accuracies. Also, we will plot the decision boundary along with the train and test samples. 
* * Next, we will use above neural network implementation to train with a single hidden layer having 5 units. Will compute the train and test set accuracies obtained by your trained model. We will now plot the decision boundary and visualize the train and test samples along with the decision boundary. Will compare the results with the ones obtained using a logistic regression classifier.
* * Next we will vary the number of units in the hidden layer as (1,2,3,10,20,40). Note that we are  still working with a single hidden layer architecture. We will report the train and test set accuracies in each case. Also, we will plot the decision boundary along with the test samples and report how does the decision boundary change as we increase the number of hidden units.
* * Next we will train a network with two hidden layers each having 5 units. Again we will report the train and test accuracies and plot the decision boundary along with test samples. Will compare results with accuracies obtained in the above tasks.

* Now We will create a network with a single hidden layer having 100 units. We will vary the learning rate(n) being inversely proportional to number of iterations(t) i.e where t is the current learning iteration number. Choose an appropriate stopping criterion based
on the change in value of the error function.
* * We will Use sigmoid as the activation function in our network.
* * Next we will use ReLU as the activtion instead of the sigmoid function. ReLU is defined using the function: g(z) = max(0; z).we will still use sigmoid in the final (output) layer since we are dealing with Boolean output.
* * Will Plot the decision boundary along with the test samples.
* * Will Report and plot the accuracy on the training and the test sets, time taken to train the network. 

# Author 
 * [Harshit Monish](https://github.com/harshitmonish)
 
## Course Project Under [Prof. Parag Singla](http://www.cse.iitd.ernet.in/~parags/)
