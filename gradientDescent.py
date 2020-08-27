#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:23:32 2020

@author: raghav
"""

import numpy

def sigmoid(sop):
    return 1.0/(1+numpy.exp(-1*sop))

def error(predicted, target):
    return numpy.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def sigmoid_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate*grad

x = 0.1
target = 0.3

learning_rate = 0.01

w = numpy.random.rand()

print("Initial W : ", w)

for k in range(1000000): 
    # Forward Pass
    y = (w*x)
    predicted = sigmoid(y)
    err = error(predicted, target)
    
    # Backward Pass
    g1 = error_predicted_deriv(predicted, target)

    g2 = sigmoid_sop_deriv(y)
    
    g3 = sop_w_deriv(x)

    grad = g3*g2*g1
    
    w = update_w(w, grad, learning_rate)

print(predicted)