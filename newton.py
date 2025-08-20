import numpy as np


def first_der(function, x, epsilon):
    """
    first_der uses forward differences to calculate the first derivative at a point

    @param x: point at which to take derivative
    @param function: univariate function to optimize
    @param epsilon: used for caculating derivatives
    @return: the forward differences-estimated first derivative
    """
    num = function(x + epsilon) - function(x)
    sign = np.sign(num)
    return sign * np.exp(np.log(abs(num)) - np.log(epsilon))


def second_der(function, x, epsilon):
    """
    second_der uses forward differences to calculate the second derivative at a point

    @param x: point at which to take derivative
    @param function: univariate function to optimize
    @param epsilon: used for caculating derivatives
    @return: the forward differences-estimated second derivative
    """
    num = first_der(function, x + epsilon, epsilon) - first_der(
        function, x, epsilon
    )  # numerator
    sign = np.sign(num)
    return sign * np.exp(np.log(abs(num)) - np.log(epsilon))


def optimize(x_cur, function, epsilon=1e-5, diff=1e-9, stop_iter=5000):
    """
    optimize runs Newton's method on a provided function, with a provided start value

    @param x_cur: start point
    @param function: univariate function to optimize
    @param epsilon: used for caculating derivatives
    @param diff: the maximum between successive x values to stop running
    @param stop_iter: the maximum iterations before stopping
    @return: the extreme value found by the method
    """
    x_prev = -float("inf")  # starting x_prev that won't violate tolerance
    iter = 0
    while (abs(x_cur - x_prev) > diff) & (
        iter < stop_iter
    ):  # while we haven't got under our tolerance
        x_prev = x_cur
        print(x_prev)
        first = first_der(function, x_prev, epsilon)
        sign_first = np.sign(first)
        second = second_der(function, x_prev, epsilon)
        sign_second = np.sign(second)
        x_cur = (
            x_prev
            - (np.exp(np.log(abs(first)) - np.log(abs(second))))
            * sign_first
            * sign_second
        )
        iter += 1
    if iter >= stop_iter:
        print("Maximum iterations exceeded! Oh no!")
        return None
    return x_cur
