import numpy as np
import copy
import itertools
from basis import Basis
from numpy.linalg import matrix_power

from qr_solver import R

def concat(a:np.array, b:np.array):
    if b.size == 0:
        return a
    if a.size == 0:
        return a
    else:
        return np.concatenate(a, b)
        
class Solution():

    def __init__(self, n_dims: int, dim_sizes: np.array, area_lims: np.array, power:int, basis: Basis) -> None:
        '''
        initiation of solution
        init of grid of cells, initial coefs, basis
        area_lims - np.array of shape (n_dims, 2) - pairs of upper and lower limits of corresponding dim
        dim_sizes - np. array of shape (n_dims)
        '''

        self.area_lims = area_lims
        self.n_dims = n_dims # = len(dim_sizes)
        self.dim_sizes = dim_sizes # n of steps for all directions 
        self.init_grid()
        self.steps = (self.area_lims / self.dim_sizes)
        self.power = power
        self.basis = basis(steps=self.steps)

    def init_grid(self) -> None:
        cells_shape = tuple([self.dim_sizes] + [self.power]*self.n_dims)
        self.cells_coefs = np.zeros(cells_shape)
        
    def localize(self, global_point: np.array, cells_closed_right: bool = False) -> np.array:
        steps = self.steps #TODO refactor
        if cells_closed_right:
            shift = np.array((global_point % steps) < 1e-12, dtype=int)
            cell_num = tuple(np.floor(global_point / steps)) - shift
        else:
            cell_num = tuple(np.floor(global_point / steps))

        #local_point = copy.deepcopy()
        local_point = global_point/steps - (cell_num - 0.5) #TODO CHECK
        return cell_num, local_point

    def globalize(self, cell_num: np.array, local_point: np.array) -> np.array:
        steps = self.steps
        global_point = (local_point + cell_num - 0.5) * steps#TODO CHECK
        return global_point


    def eval(self, global_point: np.array, derivatives: np.array, cells_closed_right: bool = False) -> np.float:
        '''
        x - np.array(n_dim, float)
        derivatives - np.array(n_dim, int)
        evaluation of solution function with argument x and list of partial derivatives
        '''
        # if (derivatives < 0).any() :
        #     raise Exception("Negative derivative")
        derivative = np.abs(derivative)

        #eval cell, that x is in
        #cell_num = tuple(np.floor(global_point / (self.area_lims / self.dim_sizes)))
        cell_num, local_point = self.localize(global_point, cells_closed_right)
        coefs = self.cell_coefs[cell_num]
        result = copy.deepcopy(coefs)
        #applying coefs tensor to evaled basis in point
        basis_evaled = np.array([self.basis.eval(coordinate, derivative) 
                                    for coordinate, derivative 
                                    in zip(local_point, derivatives)]) #TODO check
        for i in range(self.n_dim):
            result = coefs @ basis_evaled[i]
        return result

    def eval_loc(self, cell_num, local_point, derivatives):
        coefs = self.cell_coefs[cell_num]
        result = copy.deepcopy(coefs)
        #applying coefs tensor to evaled basis in point
        basis_evaled = np.array([self.basis.eval(coordinate, derivative) 
                                    for coordinate, derivative 
                                    in zip(local_point, derivatives)]) #TODO check
        for i in range(self.n_dim):
            result = coefs @ basis_evaled[i]
        return result

    def generate_system(self, cell_num: np.array, points: np.array) -> tuple(np.array, np.array):
        colloc_points, connect_points, border_points = points

        colloc_mat, colloc_r = self.generate_colloc(colloc_points)
       
        left_borders = cell_num == np.zeros(self.n_dims)
        right_borders = cell_num == self.dim_sizes
        
        left_border_for_use = np.array([np.logical_and(point == -1, left_borders).any() for point in border_points])
        right_border_for_use = np.array([np.logical_and(point == 1, right_borders).any() for point in border_points])
        border_points_for_use = border_points[np.logical_or(left_border_for_use, right_border_for_use)] 
        
        border_mat, border_r = self.generate_border(border_points_for_use)
        
        left_connect_for_use = np.array([np.logical_and(point == -1, ~left_borders).any() for point in connect_points])
        right_connect_for_use = np.array([np.logical_and(point == 1, ~right_borders).any() for point in connect_points])
        connect_points_for_use = connect_points[np.logical_or(left_connect_for_use, right_connect_for_use)] 

        connect_mat, connect_r = self.generate_connect(connect_points_for_use)

        res_mat = concat(concat(colloc_mat, border_mat), connect_mat)
        res_right = concat(concat(colloc_r, border_r), connect_r)

        return res_mat, res_right

    def iterate_cells(self, points) -> None:
        inds = [list(range(size)) for size in self.dim_sizes]
        all_cells = list(itertools.product(*inds))
        cell_shape = tuple([self.power]*self.n_dims)
        for cell in all_cells:
            mat, right = self.generate_system(cell, points)
            self.cells_coefs[cell] = np.solve(mat, right).reshape(cell_shape)

    def generate_eq(self, cell_num, left_side_operator, right_side_operator, points): #TODO refactor u_loc, u_bas, u_nei choice
        '''
        basic func for generating equation
        '''
        def left_side(operator, cell_num, point: np.array) -> np.array:
            '''must return row of coeficient for LSE'''
            x = copy.deepcopy(point)
            u_loc = lambda x, der: self.eval_loc(cell_num, x, der)   # for linearization purpses
            u_bas = self.basis.eval
            return operator(u_loc, u_bas, x) #u(x, 1)

        def right_side(operator, cell_num, point: np.array) -> np.float:

            def dir(point: np.array) -> np.array:
                direction = (np.abs(point) == 1) * point 
                return direction

            global_point = self.globalize(cell_num, point)
            x = global_point
            u_loc = lambda x, der: self.eval_loc(cell_num, point, der)
            u_nei = lambda x, der: self.eval_loc(cell_num + dir(point), point, der) #neighbour cell for connection eqs
            #u = lambda point, der: self.eval_loc(cell_num, point, der)
            return operator(u_loc, u_nei, x) #x
        
        mat = np.array([left_side(left_side_operator, cell_num,  point) for point in points]) # TODO refactor because slow
        r_side = np.array([right_side(right_side_operator, cell_num, point) for point in points])
        return mat, r_side


    def generate_colloc(self, points: np.array) -> tuple(np.array, np.array):
        colloc_left_operator = lambda _, u, x: u(x,[1])
        colloc_right_operator = lambda u, _, x: x
        colloc_mat, colloc_r = self.generate_eq(colloc_left_operator, colloc_right_operator, points) 
        return colloc_mat, colloc_r
    
    def generate_border(self, points: np.array) -> tuple(np.array, np.array):
        border_left_operator = [lambda _, u, x: u(x,[0]), lambda _, u, x: u(x,[1])]
        border_right_operator = [lambda u, _, x: 0, lambda u, _, x: 0]
        border_mat, border_r = self.generate_eq(border_left_operator[0], border_right_operator[0], points)
        for i in range(1,len(border_left_operator)):
            border_mat_small, border_r_small = self.generate_eq(border_left_operator, border_right_operator, points)
            border_mat = np.concatenate(border_mat, border_mat_small)
            border_r = np.concatenate(border_r, border_r_small)

        return border_mat, border_r

    def generate_connect(self, points: np.array) -> tuple(np.array, np.array):
        def dir(point: np.array) -> np.array:
            direction = (np.abs(point) == 1) * point 
            return direction

        # connect_left_operator = [lambda u, x: u(x) + np.sum(dir(x))*u(x, np.abs(dir(x))),
        #                          lambda u, x: u(x, 2*np.abs(dir(x))) + np.sum(dir(x)) * u(x, 3*np.abs(dir(x)))]
        # connect_right_operator = [lambda u, x: u(x-2*dir(x)) - np.sum(dir(x))*u(x-2*dir(x), np.abs(dir(x))),
        #                          lambda u, x: u(x) + np.sum(dir(x))*u(x,dir)]
        connect_left_operator = [lambda _, u, x: u(x) + np.sum(dir(x))*u(x, dir(x)),
                                 lambda _, u, x: u(x, 2*dir(x)) + np.sum(dir(x)) * u(x, 3*dir(x))]
        connect_right_operator = [lambda _, u, x: u(x-2*dir(x)) - np.sum(dir(x))*u(x-2*dir(x), dir(x)),
                                 lambda _, u, x: u(x-2*dir(x), 2*dir(x)) + np.sum(dir(x))*u(x-2*dir(x),3*dir(x))]
        
        connect_mat, connect_r = self.generate_eq(connect_left_operator[0], connect_right_operator[0], points)
        for i in range(1,len(connect_left_operator)):
            connect_mat_small, connect_r_small = self.generate_eq(connect_left_operator, connect_right_operator, points)
            connect_mat = np.concatenate(connect_mat, connect_mat_small)
            connect_r = np.concatenate(connect_r, connect_r_small)

        return connect_mat, connect_r
    # def generate_right_side(self, global_point):



