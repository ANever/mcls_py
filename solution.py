from threading import local
import numpy as np
import copy
import itertools
from basis import Basis

from qr_solver import QR_solve

import matplotlib.pyplot as plt
from matplotlib import cm

def concat(a:np.array, b:np.array):
    if b.size == 0:
        return a
    if a.size == 0:
        return a
    else:
        return np.concatenate((a, b))

def f_collocation_points(N):
    points = np.zeros(N+1)
    h = 2/(N+1)
    points[0] = -1 + h/2
    for i in range(1, N+1):
        points[i] = points[i-1] + h
    return np.array(points).reshape(N+1,1)

class Solution():

    def __init__(self, n_dims: int, dim_sizes: np.array, area_lims: np.array, power:int, basis: Basis) -> None:
        '''
        initiation of solution
        init of grid of cells, initial coefs, basis
        area_lims - np.array of shape (n_dims, 2) - pairs of upper and lower limits of corresponding dim
        dim_sizes - np. array of shape (n_dims)
        '''

        self.area_lims = np.array(area_lims)
        self.n_dims = n_dims # = len(dim_sizes)
        self.dim_sizes = np.array(dim_sizes) # n of steps for all directions 
        self.power = power
        self.init_grid()
        self.steps = ((self.area_lims[:,1] - self.area_lims[:,0]) / self.dim_sizes)
        
        self.basis = Basis(power, steps=self.steps, n_dims = n_dims)

    def init_grid(self) -> None:
        self.cells_shape = tuple(list(self.dim_sizes) + [self.power]*self.n_dims)
        self.cells_coefs = np.ones(self.cells_shape) * 0.4
        self.cell_size = self.power**self.n_dims
        
    def localize(self, global_point: np.array, cells_closed_right: bool = False) -> np.array:
        if cells_closed_right:
            shift = np.array((global_point % self.steps) < 1e-12, dtype=int)
            cell_num = np.array(np.floor(global_point / self.steps) - shift, dtype=int)
        else:
            cell_num = np.array(np.floor(global_point / self.steps), dtype=int)

        local_point = 2 * (np.array(global_point)/np.array(self.steps) - (cell_num + 0.5))

        return cell_num, local_point

    def globalize(self, cell_num: np.array, local_point: np.array) -> np.array:
        global_point = (np.array(local_point) + 2*np.array(cell_num) + 1) * self.steps/2
        return global_point


    def eval(self, point: np.array, derivatives: np.array, cell_num = None, local = False, cells_closed_right: bool = False) -> np.float:
        '''
        x - np.array(n_dim, float)
        derivatives - np.array(n_dim, int)
        evaluation of solution function with argument x and list of partial derivatives
        '''
        derivatives = np.abs(derivatives)
        
        if local:
            local_point = point
        else:
            cell_num, local_point = self.localize(point, cells_closed_right)
        
        coefs = self.cells_coefs[tuple(cell_num)]
        result = copy.deepcopy(coefs)
        #applying coefs tensor to evaled basis in point
        basis_evaled = self.basis.eval(local_point, derivatives, ravel=False)
        for b_e in basis_evaled[::-1]:
            result = result @ b_e
        return result

    def generate_system(self, cell_num: np.array, points: np.array, colloc_ops, border_ops, connect_ops = []) -> tuple:
        colloc_points, connect_points, border_points = points

        def dir(point: np.array) -> np.array:
            direction = (np.abs(point) == 1) * (np.sign(point)) 
            return direction

        w = (self.steps[0]/2)#weight
        #default connection
        if len(connect_ops) == 0:
            connect_left_operators = [lambda _, u_bas, x, x_loc: u_bas(0*dir(x_loc)) + np.sum(dir(x_loc)) * u_bas(dir(x_loc)) * w,
                                 lambda _, u_bas, x, x_loc: u_bas(2*dir(x_loc))* w**2 + np.sum(dir(x_loc)) * u_bas(3*dir(x_loc))* w**3]
            connect_right_operators = [lambda _, u_nei, x, x_loc: u_nei(0*dir(x_loc)) + np.sum(dir(x_loc))*u_nei(dir(x_loc))* w,
                                        lambda _, u_nei, x, x_loc: u_nei(2*dir(x_loc))* w**2 + np.sum(dir(x_loc)) * u_nei(3*dir(x_loc))* w**3]
            connect_ops = [connect_left_operators, connect_right_operators]

        #default colloc points
        if len(colloc_points) == 0:
            colloc_points = f_collocation_points(self.power)
        
        connect_left_operators, connect_right_operators = connect_ops

        colloc_left_operators, colloc_right_operators = colloc_ops

        border_left_operators, border_right_operators = border_ops

        colloc_mat, colloc_r = self.generate_subsystem(colloc_left_operators, colloc_right_operators, cell_num, colloc_points)
        
        left_borders = cell_num == np.zeros(self.n_dims)
        right_borders = cell_num == (self.dim_sizes-1)
        
        left_border_for_use = np.array([np.logical_and(point == -1, left_borders).any() for point in border_points])
        right_border_for_use = np.array([np.logical_and(point == 1, right_borders).any() for point in border_points])
        border_points_for_use = border_points[np.logical_or(left_border_for_use, right_border_for_use)]

        border_mat, border_r = self.generate_subsystem(border_left_operators, border_right_operators, cell_num, border_points_for_use)
        
        left_connect_for_use = np.array([np.logical_and(point == -1, ~left_borders).any() for point in connect_points])
        right_connect_for_use = np.array([np.logical_and(point == 1, ~right_borders).any() for point in connect_points])
        connect_points_for_use = connect_points[np.logical_or(left_connect_for_use, right_connect_for_use)] 
        # print('border', border_points_for_use, '\n connect', connect_points_for_use)
        
        connect_mat, connect_r = self.generate_subsystem(connect_left_operators, connect_right_operators, cell_num, connect_points_for_use)
        connect_weight = 1
        #print('colloc ', colloc_mat,'\nborder ', border_mat,'\nconnect ', connect_mat)
        res_mat = concat(concat(colloc_mat, border_mat), connect_mat * connect_weight)
        res_right = concat(concat(colloc_r, border_r), connect_r * connect_weight)

        # print(res_mat,'\n-------\n', res_right, '\n\n')
        return res_mat, res_right

    def iterate_cells(self, **kwargs) -> None:
        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        cell_shape = tuple([self.power]*self.n_dims)
        # new_cell_coefs = copy.deepcopy(self.cells_coefs)
        for cell in all_cells:
            mat, right = self.generate_system(cell, **kwargs)
            self.cells_coefs[cell] = QR_solve(mat, right).reshape(cell_shape)
            # new_cell_coefs[cell] = QR_solve(mat, right).reshape(cell_shape)
        # self.cells_coefs = new_cell_coefs

    def solve(self, threshold = 1e-10, max_iter = 10000, verbose=False, **kwargs) -> None:
        prev_coefs = copy.deepcopy(self.cells_coefs)
        for i in range(max_iter):
            self.iterate_cells(**kwargs)
            if np.max(np.abs((prev_coefs - self.cells_coefs))) < threshold:
                break
            prev_coefs = copy.deepcopy(self.cells_coefs)
        if verbose:
            print('Iterations to converge: ', i)

    def generate_eq(self, cell_num, left_side_operator, right_side_operator, points):
        '''
        basic func for generating equation
        '''
        def left_side(operator, cell_num, point: np.array) -> np.array:
            '''must return row of coeficient for LSE'''
            loc_point = copy.deepcopy(point)
            global_point = self.globalize(cell_num, point)
            x = copy.deepcopy(global_point)
            u_loc = lambda der: self.eval(loc_point, der, local = True, cell_num = cell_num)   # for linearization purpses
            u_bas = lambda der: self.basis.eval(loc_point, der, ravel=True)
            return operator(u_loc, u_bas, x, loc_point)

        def right_side(operator, cell_num, point: np.array) -> np.float:
            
            def dir(point: np.array) -> np.array:
                direction = (np.abs(point) == 1) * (np.sign(point)) 
                return direction

            global_point = self.globalize(cell_num, point)
            x = global_point
            loc_point = copy.deepcopy(point)            
            u_loc = lambda der: self.eval(loc_point, der, local = True, cell_num = cell_num)   # for linearization purpses

            neigh_point = loc_point-2*dir(loc_point)

            #print('Cell:', cell_num, ' point:' , point, ' Neigh_cell:', cell_num + dir(loc_point))

            # u_nei = lambda der: self.eval_loc(cell_num + dir(loc_point), neigh_point, der) #neighbour cell for connection eqs
            u_nei = lambda der: self.eval(neigh_point, der, local = True, cell_num = cell_num + dir(loc_point)) 
            return operator(u_loc, u_nei, global_point, loc_point) #x
        
        mat = np.zeros((len(points), self.cell_size))
        r_side = np.zeros((len(points)))
        for i in range(len(points)):
            mat[i] = left_side(left_side_operator, cell_num,  points[i])
            r_side[i] = right_side(right_side_operator, cell_num, points[i])
        return mat, r_side

    def generate_subsystem(self, left_ops, right_ops, cell_num, points: np.array) -> tuple:
        mat, r = self.generate_eq(cell_num, left_ops[0], right_ops[0], points)
        #print(mat)
        for i in range(1,len(left_ops)):
            mat_small, r_small = self.generate_eq(cell_num, left_ops[i], right_ops[i], points)
            mat = concat(mat, mat_small)
            r = concat(r, r_small)
        return mat, r
    
    def plot(self, n = 1000):
        func = np.zeros(n)
        grid = np.linspace(self.area_lims[0,0], self.area_lims[0,1], n, endpoint=False)
        for i in range(len(grid)): 
            func[i] = self.eval(grid[i], [0])
        plt.plot(func)
        plt.show()
    
    def plot2d(self, n=100):
        func = np.zeros((n,n))
        ax1 = np.linspace(self.area_lims[0,0], self.area_lims[0,1], n, endpoint=False)
        ax2 = np.linspace(self.area_lims[1,0], self.area_lims[1,1], n, endpoint=False)
        X, Y = np.meshgrid(ax1, ax2)

        for i in range(n):
            for j in range(n): 
                func[i, j] = self.eval([ax1[i], ax2[j]], [0,0])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, func, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()





#______________________________TESTING________________________

if __name__ == '__main__':
    
    def f_collocation_points(N):
        points = np.zeros(N+1)
        h = 2/(N+1)
        points[0] = -1 + h/2
        for i in range(1, N+1):
            points[i] = points[i-1] + h
        return np.array(points).reshape(N+1,1)
    colloc_points = f_collocation_points(4)


    power = 5
    params = {
        'n_dims': 1,
        'dim_sizes': np.array([5]),
        'area_lims': np.array([[0,1]]),
        'power': power,
        'basis': Basis(power),
    }
    sol = Solution(**params)

    w = (sol.steps[0]/2)

    colloc_left_operators = [lambda u_loc, u_bas, x, x_loc: u_bas([4]) * (w**4)]
    colloc_right_operators = [lambda u_loc, u_nei, x, x_loc: np.exp(x)*(x**4 + 14*(x**3) + 49*(x**2) + 32*x - 12) * (w**4)]
    colloc_ops = [colloc_left_operators, colloc_right_operators]

    border_left_operators = [lambda _, u_bas, x, x_loc: u_bas([0]), 
                                lambda _, u_bas, x, x_loc: u_bas([1]) * w]
    border_right_operators = [lambda u, _, x, x_loc: 0,
                                lambda u, _, x, x_loc: 0 * w]
    border_ops = [border_left_operators, border_right_operators]

    connect_points = np.array([[-1], [1]])
    border_points = connect_points

    points = (colloc_points, connect_points, border_points)

    iteration_dict = {'points':points,
                    'colloc_ops':colloc_ops,
                    'border_ops':border_ops}

    sol.iterate_cells(**iteration_dict)