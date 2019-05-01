#!/usr/bin/env python
"""
Created on Thu Apr 26 00:21:40 2018

@author: harshitm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:59:49 2018

@author: harshitm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#from random import choice
#import time
#import _pickle as cpickle
#import os
import time,sys,statistics,csv
import random

def normalise_data(x):
    normalized_arr = (x - np.mean(x)) / np.std(x) 
    return normalized_arr

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def sigmoid_prime(x):
    return (x * (1 - x)) 

def reLU(x):
    return x* (x > 0)

def dReLU(x):
    return 1. * (x > 0)
class network(object):
    def __init__(self, list_of_num, batch_size):
        self.num_layers = len(list_of_num)
        self.list_of_num = list_of_num
        self.batch_size = batch_size
        self.biases = [np.random.uniform(-1, 1,size = (1,y)) for y in list_of_num[1:]]
        self.weights = [np.random.uniform(-1, 1, size = (x,y)) for x,y in zip(list_of_num[:-1], list_of_num[1:])]
        #self.biases = [np.random.randn(1, y) for y in list_of_num[1:]]
        #self.weights = [np.random.randn(x, y) for x, y in zip(list_of_num[:-1], list_of_num[1:])]
        
    def forwardfeed(self, a):
        for b, w in zip(self.biases, self.weights):
            if(b.all() == self.biases[-1].all()):
                
                a = sigmoid(np.dot(a, w) + b)
            else:
                a = reLU(np.dot(a, w) + b)
        return a
    
    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        n = len(x)
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w) + b
            zs.append(z)
            if(b.all() == self.biases[-1].all()):
                activation = sigmoid(z)
            else:
                activation = reLU(z)
            activations.append(activation)
        output = activations[-1]
        error = y - output
        err_mean = np.sum(error) / error.shape[0]
        slope_output_layer = sigmoid_prime(output)
        
        delta_o = error * slope_output_layer
        #delta_o =  error / n
        grad_b[-1] =  np.sum(delta_o, axis = 0, keepdims =True)
        grad_w[-1] =  np.dot(activations[-2].T, delta_o)

        
        delta_hid = delta_o
        for i in range(2, self.num_layers):
            z = zs[-i]
            hid_err = np.dot(delta_hid, self.weights[-i + 1].transpose())
            slope_hidden_layer =  dReLU(activations[-i])
            delta_hid = hid_err * slope_hidden_layer
            #print(delta_hid, delta_o)
            grad_b[-i] =  np.sum(delta_hid, axis = 0, keepdims =True)
            grad_w[-i] =  np.dot(activations[-i -1].transpose(), delta_hid)

        return (grad_b, grad_w)
    
    def calc_cost_func(self, x, y):
        pred = self.forwardfeed(x)
        y = np.reshape(y,(len(y), 1))
        error = y - pred
        cost = np.dot(error.T, error)/ 2
        return cost

    def calc_cost_new(self, x, y):
        pred = self.forwardfeed(x)
        temp1 = np.ones(pred.shape)
        pred_def = temp1 - pred
        temp2 = np.ones(y.shape)
        y_def = temp2 - y
        error = 0
        for i in range(0,y.shape[0]):
            error += (y[i]*np.log(pred[i])) + ((y_def[i])*(np.log(pred_def[i])))
        #error = np.dot(y.T, np.log(pred))
        #error += np.dot(y_def.T, pred_def)
        return -error    
        
    def SGD(self, training_data):
        train_len = len(training_data)
        iter = 100
        eita = math.exp(-9)
        eta = 0.006
        print(eita)
        mini_batch_size = self.batch_size
        delta_grad_b = [np.zeros(b.shape) for b in self.biases]
        delta_grad_w = [np.zeros(w.shape) for w in self.weights]
        delta_grad_b_avg = [np.zeros(b.shape) for b in self.biases]
        delta_grad_w_avg = [np.zeros(w.shape) for w in self.weights] 
        converged = False
        cost_arr = []
        cost_arr.append(self.calc_cost_func(training_data[:,:-1], training_data[:,-1]))
        count  = 0
        while(converged == False):
            random.shuffle(training_data)
            print(count)
            count+= 1
            #eta = 3 / math.sqrt(count)
            if mini_batch_size != training_data.shape[0]:
                num_iter = int(train_len/mini_batch_size)
                
                #for i in range(0,num_iter):
                    #mini_batch = np.random.randint(train_, size = mini_batch_size)
                    #random.shuffle(training_data)
                mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, train_len, mini_batch_size)]
                #print(len(mini_batches))
                for mini_batch in mini_batches:
                    #print(mini_batch)
                    delta_grad_b, delta_grad_w = self.update_mini_batch(mini_batch, eta)
                    delta_grad_b_avg += delta_grad_b
                    delta_grad_w_avg += delta_grad_w
                #delta_grad_b_avg = delta_grad_b_avg / num_iter
                #delta_grad_w_avg = delta_grad_w_avg / num_iter
                #delta_grad_b_avg = [b/num_iter for b in delta_grad_b_avg ]
                #delta_grad_w_avg = [w/num_iter for w in delta_grad_w_avg]
                #self.weights = [w + (eta)*gw for w, gw in zip(self.weights, delta_grad_w_avg)]
                #self.biases = [b + (eta)*gb for b, gb in zip(self.biases, delta_grad_b_avg)]
                cost = self.calc_cost_func(training_data[:,:-1], training_data[:,-1])
                if(abs(cost - cost_arr[-1]) < eita):
                    converged = True
                else:
                    cost_arr.append(cost)
                print(cost)
            else:
                delta_grad_b, delta_grad_w = self.update_mini_batch(training_data, eta)
                self.weights = [w + (eta)*gw for w, gw in zip(self.weights, delta_grad_w)]
                self.biases = [b + (eta)*gb for b, gb in zip(self.biases, delta_grad_b)]
                cost = self.calc_cost_new(training_data[:,:-1], training_data[:,-1])
                if(cost == cost_arr[-1]):
                    converged = True
                else:
                    cost_arr.append(cost)
                print(cost)
        #print(self.weights, self.biases)
    def update_mini_batch(self, mini_batch, eta):
        x_new = mini_batch[:,:-1]
        y = mini_batch[:,-1]
        y_new = np.reshape(y,(len(y), 1))
        #print(y_new.shape)
        delta_grad_b, delta_grad_w = self.backprop(x_new, y_new)
        #return delta_grad_b, delta_grad_w
        self.weights = [w + (eta)*gw for w, gw in zip(self.weights, delta_grad_w)]
        self.biases = [b + (eta)*gb for b, gb in zip(self.biases, delta_grad_b)]
        return delta_grad_b, delta_grad_w

    
    def evaluate(self, testX):
        results = []
        test_results = self.forwardfeed(testX)
        #print(test_results)
        for i in test_results:
            #print(i[0]) 
            if i[0] > 0.5:
                results.append(1.0)
            else:
                results.append(0.0)
        #print(results)
        return results
        
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

def get_data(data_mat):
    N, M = data_mat.shape
    x = np.zeros((N,M-1), dtype = int)
    y = np.zeros(N, dtype = int)
    x = data_mat[:,0:M-1]
    y = data_mat[:,M-1]
    for i in range(0, M-1):
        x[i] = x[i] / 255
    return x, y   

def find_accuracy(results, y):
    sum_ = 0
    len_ = len(results)
    for i in range(0,len_):
        if results[i] == y[i]:
            sum_ += 1
    accuracy = sum_/len_
    print("accuracy : ", sum_/len_)
    return accuracy

def train_lib_svm():
    y_train, x_train = svm_read_problem("train_ass4_c.csv")
    y_test, x_test = svm_read_problem("test_ass4_c.csv")
    m = svm_train(y_train, x_train, '-c 1')
    res, acc, val = svm_predict(y_test ,x_test, m)
    print("accuracy is : ", val)
    
def main():
    train_data = pd.read_csv("train_ass4_c.csv",header=None)
    test_data = pd.read_csv("test_ass4_c.csv",header=None)
   # print(train_data)
    #relevant_data = train_data.loc[train_data[784].isin([6,8])]
    #relevant_data.to_csv("train_ass4_c.csv", encoding='utf-8', index=False)
    #relevant_data_test = test_data.loc[test_data[784].isin([6,8])]
    #print(relevant_data_test.values.shape)
    #relevant_data_test.to_csv("test_ass4_c.csv", encoding='utf-8', index=False)
    #train_rel = get_relevant_data(train_data.values)
    #test_rel = get_relevant_data(test_data.values)
    x_train, y_train = get_data(train_data.values)
    x_test, y_test = get_data(test_data.values)
    list_of_num = [784,100,1]
    training_data = np.c_[x_train, y_train]
    batch_size = 100
    #batch_size = x_train.shape[0]
    net = network(list_of_num, batch_size)
    net.SGD(training_data)
    results_train = net.evaluate(x_train)
    #print(results_train)
    #print(y_train)
    results_test = net.evaluate(x_test)
    accuracy_train = find_accuracy(results_train, y_train)
    accuracy_test = find_accuracy(results_test, y_test)
    #train_lib_svm()
    
if __name__ == "__main__":
    main()
