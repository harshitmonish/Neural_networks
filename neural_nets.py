# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:17:15 2018

@author: harshitm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import math
#from random import choice
#import time
#import _pickle as cpickle
#import os
import time,sys,statistics,csv
import random
from sklearn.linear_model import LogisticRegression

def normalise_data(x):
    normalized_arr = (x - np.mean(x)) / np.std(x) 
    return normalized_arr

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def sigmoid_prime(x):
    return (x * (1 - x)) 

class network(object):
    def __init__(self, list_of_num, batch_size):
        self.num_layers = len(list_of_num)
        self.list_of_num = list_of_num
        self.batch_size = batch_size
        self.biases = [np.random.uniform(size = (1,y)) for y in list_of_num[1:]]
        self.weights = [np.random.uniform(size = (x,y)) for x,y in zip(list_of_num[:-1], list_of_num[1:])]
        #self.biases = [np.random.randn(1, y) for y in list_of_num[1:]]
        #self.weights = [np.random.randn(x, y) for x, y in zip(list_of_num[:-1], list_of_num[1:])]
        
    def forwardfeed(self, a):
        
        #print(a.shape, len(self.biases), len(self.weights))
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(a, w) + b)
        return a
    
    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        output = activations[-1]
        error = y - output
        err_mean = np.sum(error) / error.shape[0]
        slope_output_layer = sigmoid_prime(output)
        
        delta_o = error * slope_output_layer
        grad_b[-1] = np.sum(delta_o, axis = 0, keepdims =True)
        grad_w[-1] = np.dot(activations[-2].T, delta_o)

        
        delta_hid = delta_o
        for i in range(2, self.num_layers):
            z = zs[-i]
            hid_err = np.dot(delta_hid, self.weights[-i + 1].transpose())
            slope_hidden_layer = sigmoid_prime(activations[-i])
            delta_hid = hid_err * slope_hidden_layer
            grad_b[-i] = np.sum(delta_hid, axis = 0, keepdims =True)
            grad_w[-i] = np.dot(activations[-i -1].transpose(), delta_hid)

        return (grad_b, grad_w)
    
    def SGD(self, training_data):
        train_len = len(training_data)
        iter = 999
        eta = 0.09
        mini_batch_size = self.batch_size

        
        for j in range(0,iter):
            if mini_batch_size != training_data.shape[0]:
                random.shuffle(training_data)
                mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, train_len, mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
            else:
                self.update_mini_batch(training_data, eta)
    def update_mini_batch(self, mini_batch, eta):
        x_new = mini_batch[:,:-1]
        y = mini_batch[:,-1]
        y_new = np.reshape(y,(len(y), 1))
        #print(y_new.shape)
        delta_grad_b, delta_grad_w = self.backprop(x_new, y_new)
        self.weights = [w + (eta)*gw for w, gw in zip(self.weights, delta_grad_w)]
        self.biases = [b + (eta)*gb for b, gb in zip(self.biases, delta_grad_b)]
    
    def evaluate(self, testX):
        results = []
        test_results = self.forwardfeed(testX)
        for i in test_results:
            if i > 0.5:
                results.append(1.0)
            else:
                results.append(0.0)
        return results
        
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
def train_logistic(trainX, trainY, testX, testY):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(trainX, trainY)
    result_test = logisticRegr.predict(testX)
    score_test = logisticRegr.score(testX, testY)
    print("Test accuracy is : ", score_test)
    plot_decision_boundary(lambda x:logisticRegr.predict(x), trainX, trainY[:,0])
    result_train = logisticRegr.predict(trainX)
    score_train = logisticRegr.score(trainX, trainY)
    print("Train accuracy is: ", score_train)

    plot_decision_boundary(lambda x:logisticRegr.predict(x), testX, testY[:,0])
    #score_test = logisticRegr.score(trainX, trainY)
    #print("accuracy is : ", score_test)
    #plot_decision_boundary()
    
def find_accuracy(results, y):
    sum_ = 0
    len_ = len(results)
    for i in range(0,len_):
        if results[i] == y[i]:
            sum_ += 1
    accuracy = sum_/len_
    print("accuracy : ", sum_/len_)
    return accuracy
 
counter = 0
def plot_decision_boundary(model, X, y):
    """
    Given a model(a function) and a set of points(X), corresponding labels(y), 
    scatter the points in X with color coding according to y.
    Also use the model to predict the label at grid points to
    get the region for each label, and thus the descion boundary.
    Example usage:
    say we have a function predict(x,other params) which makes 0/1 prediction 
    for point x and we want to plot train set then call as:
    plot_decision_boundary(lambda x:predict(x,other params),X_train,Y_train)
    params(3): 
        model : a function which expectes the point to make 0/1 label prediction
        X : a (mx2) numpy array with the points
        y : a (mx1) numpy array with labels
    outputs(None)
    """
    global counter 
    counter += 1
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = np.array(model(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    fig = plt.figure(figsize =(10,8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')

    plt.show()
    file_name = 'Assignment3_2012VST9734\image'
    file_name += str(counter)
    file_name += '.png'
    fig.savefig(file_name)
    
def main():
    trainX = pd.read_csv("toy_data/toy_trainX.csv",header=None).values
    trainY = pd.read_csv("toy_data/toy_trainY.csv",header=None).values
    testX = pd.read_csv("toy_data/toy_testX.csv",header=None).values
    testY = pd.read_csv("toy_data/toy_testY.csv",header=None).values
    #x = np.c_[np.ones(N), x]
    
    train_data = np.c_[trainX, trainY]
    test_data = np.c_[testX, testY]
    #train_logistic(trainX, trainY, testX, testY)
    
    list_of_num = [2,5,5,1]
    batch_size = trainX.shape[0]
    net = network(list_of_num, batch_size)
    net.SGD(train_data)
    results_train = net.evaluate(trainX)
    results_test = net.evaluate(testX)
    
    accuracy_train = find_accuracy(results_train, trainY)
    accuracy_test = find_accuracy(results_test, testY)
    

    #plot_decision_boundary(lambda x:net.evaluate(x), trainX, trainY[:,0])
    #plot_decision_boundary(lambda x:net.evaluate(x), testX, testY[:,0])
    
if __name__ == "__main__":
    main()