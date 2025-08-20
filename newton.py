# implement Newton's method

import math
import numpy as np

def square(x):
    return(x**2)

def first_der(function, x, epsilon):
    return((function(x+epsilon)-function(x))/epsilon)

def second_der(function, x, epsilon):
    return((first_der(function, x+epsilon, epsilon)-first_der(function, x, epsilon))/epsilon)

def optimize(x_cur, function, epsilon = 1e-5, diff = 1e-9, stop_iter = 5000):
    x_prev = -float('inf') # starting x_prev that won't violate tolerance
    iter = 0

    while (x_cur - x_prev) > diff & iter < stop_iter: # while we haven't got under our tolerance
        x_prev = x_cur
        x_cur = x_prev - first_der(function, x_prev, epsilon)/second_der(function, x_prev, epsilon)
        iter += 1
    
    if iter >= stop_iter:
        print("Maximum iterations exceeded! Oh no!")
        return None
    
    return (x_cur)


