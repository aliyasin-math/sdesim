import numpy as np
import matplotlib.pyplot as plt

def RK(y0,N,h,F):

    xs = np.linspace(0,N*h,N+1)
    ys = np.zeros((N+1,len(y0)))

    ys[0,:] = y0

    for i in range(N):
        ks0 = h*F(xs[i],ys[i,:])
        ks1 = h*F(xs[i] + h/2,ys[i,:]+ks0/2)
        ks2 = h*F(xs[i] + h/2,ys[i,:]+ks1/2)
        ks3 = h*F(xs[i+1],ys[i,:]+ks2)
        
        ys[i+1,:] = ys[i,:] + (ks0 + 2*ks1 + 2*ks2 + ks3)/6
    return xs,ys