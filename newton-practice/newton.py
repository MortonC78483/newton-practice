from autograd import hessian, grad
import autograd.numpy as np
from scipy.linalg import solve
import numbers

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
    ) 
    
    # return 1 if we were diverging (in which case, the main function would catch it)
    # or if we passed a linear function, where the second derivative was 0.
    if num == 0:
        return 1
    else:
        return num/(2 * epsilon)

def single_variate_step(x_prev, x_cur, function, epsilon):
    x_prev = x_cur
    first = first_der(function, x_prev, epsilon)
    second = second_der(function, x_prev, epsilon)
    x_cur = (x_prev - first/second) 
    return x_prev, x_cur
    
def multi_variate_step(x_cur, function, epsilon):
    """
    multi_variate_step runs a single step of Newton's method on a provided function

    @param x_cur: current x value
    @param function: function to optimize
    @param epsilon: used for caculating derivatives
    @return: tuple of the provided x_cur and the new x value
    """
    grad_f = grad(function)
    gradient = grad_f(x_cur)

    hess_f = hessian(function)
    hess = hess_f(x_cur)

    x_new = x_cur - solve(hess, gradient)

    return x_cur, x_new
    
    
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
        raise TypeError('`function` must be a function')
    
    if isinstance(x_cur, numbers.Number):
        x_cur = np.asarray([x_cur])
        
    try:
        x_cur = np.asarray(x_cur)
    except TypeError:
        print('`x_cur` must be numeric or array-like')

    x_prev = np.copy(x_cur) + 2 * diff # starting x_prev that won't violate tolerance
    iter = 0

    # relies on the starting x_prev
    while (np.linalg.norm(x_cur - x_prev) > diff) & ( 
        iter < stop_iter
    ):  
        x_prev, x_cur = multi_variate_step(x_cur, function, epsilon)
        iter += 1

        if np.linalg.norm(x_cur) > 1e7:
            raise RuntimeError('function appears to diverge.')
            
    if iter >= stop_iter:
        print("Maximum iterations exceeded! Oh no!")
        return None
    return x_cur

    


    