import numpy as np
import info_theory as it

def test_fun():
    '''
    see if the function works
    '''

    # write a statement, we test to see if it is true
    assert it.fun(0) == np.log(3)
    