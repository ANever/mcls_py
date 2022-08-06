import copy
import numpy as np


def rotation_vec(x):
    e = np.zeros(len(x))
    e[0] = 1
    rot_vec_1 = x + np.sqrt(x.dot(x)) * e
    rot_vec_2 = x - np.sqrt(x.dot(x)) * e
    if rot_vec_1.dot(rot_vec_1) > rot_vec_2.dot(rot_vec_2):
        rot_vec = rot_vec_1
    else:
        rot_vec = rot_vec_2
        
    # if rot_vec.dot(rot_vec) < 10**(-8):
    #     return rot_vec
    # else:
    rot_vec = rot_vec / np.sqrt(rot_vec.dot(rot_vec))
    return rot_vec

def QR_housholder(A, b):
    R = copy.deepcopy(A)
    Q = np.eye(A.shape[0])
    u_0 = np.zeros(A.shape[0])
    for j in range(min(A.shape)):
            u = rotation_vec(R[j:,j])
            if j==0:
                u_0 = u
            else:
                R[j:,j-1] = u
            R[j:,j:] = R[j:,j:] - 2*np.outer(u, u.dot(R[j:,j:]))
            Q[j:,j:] = Q[j:,j:] - 2*np.outer(u, u.dot(Q[j:,j:]))
            b[j:] = b[j:] - 2*u*(u.dot(b[j:]))
    return Q, R, b, u_0

def fast_dot_Q(right_side, u_0, R):
    right_side = right_side - 2*u_0*(u_0.dot(right_side))
    for j in range(1,min(R.shape)):
        u = R[j:,j-1]
        right_side[j:] = right_side[j:] - 2*u*(u.dot(right_side[j:]))
        result = right_side
    return result

def _solve(R, b):
    length = min(R.shape)
    x = np.zeros(length)
    for i in range(length-1, -1, -1):
        for j in range(length-1, -1, -1):
            if i < j:
                b[i] -= b[j] * R[i,j] / R[j,j]
        x[i] = b[i] / R[i,i]
    return x

def QR_solve(A, b):
    Q, R, b, u_0 = QR_housholder(A, b)
    res = _solve(R, b)
    return res


#Примеры применения

#стандартный

A = np.array([[1,0,1],[0,1,1],[3,1,4], [1,1,4]])
b = np.array([1,1,0,1])

x = QR_solve(A, b)


#если есть много одинаковых матриц с разными правыми частями, можно считать быстрее

b1 = np.array([1,1,0,1])
b2 = np.array([1,2,4,1])
b3 = np.array([1,6,5,1])

Q, R, _, u_0 = QR_housholder(A, b)

b1 = fast_dot_Q(b1, u_0, R)
b2 = fast_dot_Q(b2, u_0, R)
b3 = fast_dot_Q(b3, u_0, R)

x1 = _solve(R, b1)
x2 = _solve(R, b2)
x3 = _solve(R, b3)