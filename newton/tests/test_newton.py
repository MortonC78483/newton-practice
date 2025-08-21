import pytest
import numpy as np
import math

import newton

def test_basic_function():
    assert np.isclose(newton.optimize(2.95, np.cos), math.pi)

def test_bad_input():
    #with pytest.raises(TypeError):   
    #    newton.optimize(np.cos, 2.95)
    ## Ideally, our function would raise the exception with a useful message.
    with pytest.raises(TypeError, match='`x_cur` must be numeric or array-like'):
        newton.optimize(np.cos, np.cos)
    with pytest.raises(TypeError, match='`function` must be a function'):
        newton.optimize(2.95, 2.95)
    
def test_divergence():
    def test_func(x):
        return x**4/4 - x**3 - x
    
    with pytest.raises(RuntimeError, match = 'function appears to diverge.'):
        newton.optimize(1, test_func, epsilon = 1e-6)