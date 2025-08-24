import numpy as np
import matplotlib.pyplot as plt
from sdesim.derivops.finite_diff import dxf

# Strong Order 1 Milstein Scheme

def M(z0,N,h,F,G,rng=None):

    """
    Milstein Method for multi-dim SDEs with diagonal noise only.
    F, G: callables returning vectors in R^d; G is elementwise (diagonal).
    rng: None -> fresh randomness; int or np.random.Generator -> reproducible.
    """
    rng = np.random.default_rng(rng)
    ts = np.linspace(0,N*h,N+1)
    zs = np.zeros((N+1,len(z0)))

    zs[0,:] = z0

    for i in range(N):
        dW = np.sqrt(h)*np.random.randn(len(z0))
        zs[i+1] = zs[i] + F(ts[i],zs[i])*h + G(ts[i],zs[i])*dW + 1/2 * (G(ts[i],zs[i])) * (dxf(ts[i], zs[i], G)) * (dW**2 - h)
    return ts,zs