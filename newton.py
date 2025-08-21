import numpy as np
import numbers
from autograd import hessian, grad
import autograd.numpy as np
from scipy.linalg import inv 
import math

# This is the single variate version.
def first_der(function, x, epsilon):
    """
    first_der uses forward differences to calculate the first derivative at a point

    @param x: point at which to take derivative
    @param function: univariate function to optimize
    @param epsilon: used for caculating derivatives
    @return: the forward differences-estimated first derivative
    """
    num = function(x + epsilon) - function(x - epsilon)
    return num/(2 * epsilon)

def second_der(function, x, epsilon):
    """
    second_der uses forward differences to calculate the second derivative at a point

    @param x: point at which to take derivative
    @param function: univariate function to optimize
    @param epsilon: used for caculating derivatives
    @return: the forward differences-estimated second derivative
    """
    num = first_der(function, x + epsilon, epsilon) - first_der(
        function, x - epsilon, epsilon
    )  # numerator
    if num == 0:
        return 1
    else:
        return num/(2 * epsilon)

# This is the multi variate version.
def single_variate_step(x_prev, x_cur, function, epsilon):
    x_prev = x_cur
    first = first_der(function, x_prev, epsilon)
    second = second_der(function, x_prev, epsilon)
    x_cur = (x_prev - first/second) 
    return x_prev, x_cur
    
def multi_variate_step(x_prev, x_cur, function, epsilon):
    # Compute Hessian using autograd
    hess_f = hessian(function)
    hess = hess_f(x_cur)

    # Compute gradient using autograd
    grad_f = grad(function)
    gradient = grad_f(x_cur)

    # Newton-Raphson step
    x_new = x_cur - np.matmul(inv(hess), gradient)

    return np.copy(x_cur), np.copy(x_new)
    
    #x_prev = x_cur
    #hess_f = jacobian(egrad(function))  # returns a function
    #hess = hess_f(x_cur) 
    #hess_f = hessian(function)
    #hess = hess_f(x_prev)
    #grad = first_der(function, x_prev, epsilon)
    #x_cur = x_prev - np.matmul(inv(hess), grad)
    #return x_prev, x_cur
    
    
def optimize(x_cur, function, epsilon=1e-6, diff=1e-9, stop_iter=5000):
    """
    optimize runs Newton's method on a provided function, with a provided start value

    @param x_cur: start point
    @param function: univariate function to optimize
    @param epsilon: used for caculating derivatives
    @param diff: the maximum between successive x values to stop running
    @param stop_iter: the maximum iterations before stopping
    @return: the extreme value found by the method
    """
    if not callable(function):
        raise TypeError(f'`function` must be a function')
        
    x_prev = np.copy(x_cur) + 2 * diff # starting x_prev that won't violate tolerance
    iter = 0
    
    while (np.linalg.norm(x_cur - x_prev) > diff) & (
        iter < stop_iter
    ):  # while we haven't got under our tolerance
        if isinstance(x_cur, numbers.Number): # single variate
            x_prev, x_cur = single_variate_step(x_prev, x_cur, function, epsilon)
        else: # multivariate
            print(x_prev, x_cur)
            x_prev, x_cur = multi_variate_step(x_prev, x_cur, function, epsilon)
            print(x_prev, x_cur)
        iter += 1

        if np.linalg.norm(x_cur) > 1e7:
            raise RuntimeError(f'function appears to diverge.')
            
    if iter >= stop_iter:
        print("Maximum iterations exceeded! Oh no!")
        return None
    return x_cur

    


    