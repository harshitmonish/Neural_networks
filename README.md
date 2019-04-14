# Project Title:

##  To predict poker hands using the Neural Network algorithm

### In this problem, we will work with the Poker Hand dataset available on the [UCI repository](https://archive.ics.uci.edu/ml/datasets/Poker+Hand). The training set contains 25010 examples whereas the test set contains 1000000 examples each. The dataset consists of 10 categorical attributes. The last entry in each row denotes the class label.Each record is an example of a hand consisting of five playing cards drawn from a standard deck of 52. Each card is described using two attributes (suit and rank), for a total of 10 predictive attributes. There is one Class attribute that describes the "Poker Hand". The order of cards is important, which is why there are 480 possible Royal Flush hands as compared to 4
#### Attribute Information:
#### 1) S1 "Suit of card #1" 
#### Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

#### 2) C1 "Rank of card #1" 
#### Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

#### 3) S2 "Suit of card #2" 
#### Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

#### 4) C2 "Rank of card #2" 
#### Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

* 0: Nothing in hand; not a recognized poker hand 
* 1: One pair; one pair of equal ranks within five cards 
* 2: Two pairs; two pairs of equal ranks within five cards 
* 3: Three of a kind; three equal ranks within five cards 
* 4: Straight; five cards, sequentially ranked with no gaps 
* 5: Flush; five cards with the same suit 
* 6: Full house; pair + different rank three of a kind 
* 7: Four of a kind; four equal ranks within five cards 
* 8: Straight flush; straight + flush 
* 9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush 

## Tasks Done:-

* Implemented a generic neural network architecture. Implemented the backpropagation algorithm to train the network. You should train the network using Stochastic Gradient Descent (SGD) where the batch size is an input.Used the sigmoid function as the activation unit.
* Experimented with a neural network having a single hidden layer. 
* * Varied the number of hidden layer units from the set {5, 10, 15, 20, 25}. the learning rate is 0.1. 
* * Choosen a suitable stopping criterion.
* * Reported and ploted the accuracy on the training and the test sets, time taken to train the network. 
* * Ploted the metric on the Y axis against the number of hidden layer units on the X axis. 
* * Additionally, reported the confusion matrix for the test set, for each of the above parameter values. 
* Experimented with a neural network having two hidden layers, each having the same number of neurons.
* Used ReLU as the activation instead of the sigmoid function, only in the hidden layer(s).
* * ReLU is defined using the function: g(z) = max(0, z). 
* * Correctly implemented gradient descent by making use of sub-gradient at z = 0.
