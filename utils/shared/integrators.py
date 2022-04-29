from numba import cuda,njit
from numpy import zeros

@njit(nogil=True)
def myRK4(func,y0,tSpan,paramaters):
    '''This function provides a sovler for any first order system using RK4 fixed time step algorithm.

    Inputs: func - ode function point, y0 - inital conditions in the form of a numpy array (Nx1), tSpan - time span of integration (Mx1 vector of start time and end time), 
    parameters - any parameters passed to the ode function

    Outputs: ode solution in the form of a MxN matrix

    Created: 10/12/21
    Author: Hunter Quebedeaux'''
    numTimeSteps = tSpan.size
    h = tSpan[1] - tSpan[0]
    y = zeros((numTimeSteps,y0.size))
    y[0] = y0
    
    for i in range(1,numTimeSteps):
        k1 = h*func(tSpan[i-1],y[i-1],paramaters)
        k2 = h*func(tSpan[i-1]+0.5*h,y[i-1]+0.5*k1,paramaters)
        k3 = h*func((tSpan[i-1]+0.5*h),(y[i-1]+0.5*k2),paramaters)
        k4 = h*func((tSpan[i-1]+h),(y[i-1]+k3),paramaters)
        y[i]= y[i-1] + (k1+2*k2+2*k3+k4)/6
    return y


