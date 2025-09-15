import numpy as np
import matplotlib.pyplot as plt

def EM(z0,N,h,F,G,rng=None):

    rng = np.random.default_rng(rng)
    ts = np.linspace(0,N*h,N+1)
    zs = np.zeros((N+1,len(z0)))

    zs[0,:] = z0


    for i in range(N):
        dW = np.sqrt(h)*np.random.randn(len(z0))
        zs[i+1] = zs[i] + F(ts[i],zs[i])*h + G(ts[i],zs[i])*dW
    return ts,zs