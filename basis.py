import numpy as np

class Basis():
    '''
    class of basis functions for decomposition of solution
    '''
    def __init__(self, num_of_elems: int, type: str = 'poly', steps = None):
        self.type = type
        self.n = num_of_elems
        if not steps:
            self.steps = steps
        else:
            self.steps = np.ones(self.n)

    def eval(self, x, derivative = None):
        '''
        evaluation of n-th basis funcion in x
        '''
        if not(derivative):
            derivative = np.zeros(self.n)
        else:
            derivative = np.array(derivative)

        result = np.array([x**(n) for n in range(self.n)])
        # for n in range(self.n):
        #     for i in range(derivative):
        #         result[n] *= n-i
        #         result[n] *= (self.steps[n]/2) ** i
        return result
            