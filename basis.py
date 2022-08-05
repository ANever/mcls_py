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
        # self.shape = tuple([self.n]*self.n_dims)

    #     self.init_derivative_matrix()


    # def init_derivative_matrix(self):
    #     mat = np.zeros((self.n, self.n))
    #     for i in range(self.n-1):
    #         mat[i, i+1] = i
    #     self.derivative_mat = mat

    def eval(self, x, derivative = np.array([])):
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

        return result

                # for n in range(self.n):
        #     for i in range(derivative):
        #         result[n] *= n-i
        #         result[n] *= (self.steps[n]/2) ** i
            