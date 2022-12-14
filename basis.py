import numpy as np

class Basis():
    '''
    class of basis functions for decomposition of solution
    '''
    def __init__(self, num_of_elems: int, type: str = 'poly', steps = np.array([]), n_dims: int = 1):
        self.type = type
        self.n = num_of_elems
        self.n_dims = n_dims
        if steps.size == 0:
            self.steps = np.ones(self.n)
        else:
            self.steps = steps

    def eval(self, x, derivative = np.array([]), ravel = False):
        '''
        evaluation of n-th basis funcion in x
        '''
        derivative = np.array(np.abs(derivative), dtype=int)
        if derivative.size == 0:
            derivative = np.zeros(self.n, dtype=int)

        result = np.zeros((self.n_dims, self.n))
        for i in range(self.n_dims):
            for n in range(self.n):
                mult = np.prod(list(range(max(n-derivative[i]+1,0), n+1))) / ((self.steps[i]/2)**derivative[i])
                result[i, n] = x[i]**(max(n-derivative[i], 0)) * mult
        if ravel:
            mat_result = result[0]
            for i in range(1, self.n_dims): 
                mat_result = np.outer(mat_result, result[i])
            return mat_result.ravel(order='C')
        else:
            return result
            