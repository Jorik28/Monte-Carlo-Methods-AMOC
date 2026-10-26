import numpy as np

def g(y,l,a):
    return l - abs(a-y)*y

def dg_dy(y,l,a):
    """
    derivative of g with respect to y
    """
    if a>y:
        return -a+2*y
    else:
        return -2*y + a

def root_finder(l,a,y0,N=10000,eps=1e-8):
    """"
    This function finds a root of g using Newton's method or raises an error if no root can be found
    """
    y_old = y0
    has_converged = False
    for _ in range(N):
        y_new = y_old - g(y_old,l,a)/dg_dy(y_old,l,a)
        if abs(y_old - y_new) < eps:
            has_converged = True
            break
        y_old = y_new

    if has_converged:
        return y_new
    else:
        raise ValueError(f"Newton's method did not converge for lambda={l}, a={a} and y_0={y0}.")

def y_eq(l,a,y0=1.14):
    y_root = root_finder(l,a,y0)
    if dg_dy(y_root,l,a) < 0:
        return y_root
    else:
        raise ValueError(f"No stable root found for lambda={l} and a={a}.")

def y_eq_direct(l,a,all_roots=False):
    """
    Solves lambda = |a-y|y by ABC-formula
    """
    y_candidates = []

    y1 = (a + np.sqrt(a**2 -4*l))/2
    y2 = (a - np.sqrt(a**2 -4*l))/2
    if a >= y1:
        y_candidates.append(y1)
    if a >= y2:
        y_candidates.append(y2)
    
    y3 = (a + np.sqrt(a**2 +4*l))/2
    y4 = (a - np.sqrt(a**2 +4*l))/2
    if a < y3:
        y_candidates.append(y3)
    if a < y4:
        y_candidates.append(y4)

    if all_roots:
        return y_candidates
    else:
        return max(y_candidates)


def V(y,l,a):
    return l*(a-y) -(a/2)*(a-y)*abs(a-y) + (1/3)*((a-y)**2)*abs(a-y)

def grad_V(y,l,a):
    return -l + a * np.abs(a-y) - (a-y)*np.abs(a-y)

def grad_y_eq(l,a):
    """
    Computes the gradient of the function y_eq using central finite differences
    """
    h = 0.01
    dy_dl = (y_eq(l+h,a)-y_eq(l-h,a))/(2*h)
    dy_da = (y_eq(l,a+h)-y_eq(l,a-h))/(2*h)
    return np.array([dy_dl,dy_da])