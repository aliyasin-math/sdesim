import numpy as np
import matplotlib.pyplot as plt
from sdesim.derivops.finite_diff import dtf, dttf, dxf, dxxf, dxxxf, dtxf, dtxxf


#Define the Ito Operator functions

def L0(t,z,F,G,g):
      L0 = dtf(t,z,g) + F(t,z) * dxf(t,z,g) + (1/2)*(G(t,z)**2)*dxxf(t,z,g)
      return L0 

def L1(t,z,F,G,g):
      L1 = G(t,z)*dxf(t,z,g)
      return L1

def L0_L1(t,z,F,G,g):
      L0_L1 = L0_L1 = ((dtf(t, z, G) * dxf(t, z, g) 
                    + G(t, z) * dtxf(t, z, g))
                    + F(t, z) * ( dxf(t, z, G) * dxf(t, z, g) + G(t, z) * dxxf(t, z, g))
                    + 0.5 * (G(t, z)**2) * (( dxxf(t, z, G) * dxf(t, z, g))
                    + 2.0 * dxf(t, z, G) * dxxf(t, z, g)
                    + G(t, z) * dxxxf(t, z, g)))
      return L0_L1

def L1_L0(t,z,F,G,g):
      L1_L0 = L1_L0 = G(t, z) * (dtxf(t, z, g)                                               
          + dxf(t, z, F) * dxf(t, z, g)             
          + F(t, z) * dxxf(t, z, g)                                
          + (G(t, z) * dxf(t, z, G)) * dxxf(t, z, g)                  
          + 0.5 * (G(t, z)**2) * dxxxf(t, z, g) )
      return L1_L0

def L1_L1(t,z,F,G,g):
      L1_L1 = L1_L1 = G(t, z) * ( dxf(t, z, G) * dxf(t, z, g) ) + (G(t, z)**2) * dxxf(t, z, g)
      return L1_L1

def L1_L1_L1(t,z,F,G,g):
      L1_L1_L1 = G(t, z) * (
            (dxf(t, z, G)*dxf(t, z, G) + G(t, z)*dxxf(t, z, G)) * dxf(t, z, g)
          + (3.0 * G(t, z) * dxf(t, z, G)) * dxxf(t, z, g)
          + (G(t, z)**2) * dxxxf(t, z, g))
      return L1_L1_L1

# Strong Order 2 Ito-Taylor Scheme

def ito2(z0,N,h,F,G, *,rng=None, progress=None, verbose=False, print_every=200):
    
    """
    Strong order-2 Ito–Taylor for multi-dim SDEs with diagonal noise only.
    F, G: callables returning vectors in R^d; G is elementwise (diagonal).
    rng: None -> fresh randomness; int or np.random.Generator -> reproducible.
    """
    rng = np.random.default_rng(rng)
    ts = np.linspace(0,N*h,N+1)
    zs = np.zeros((N+1,len(z0)))

    zs[0,:] = z0

    for i in range(N):
        dW = np.sqrt(h)*np.random.randn(len(z0))
        dZ = np.sqrt((h**3)/3)*np.random.randn(len(z0))

        # Define the Ito Coefficients
        I_00  = 0.5 * h**2
        I_01  = 0.5 * (h * dW + dZ)
        I_10  = 0.5 * (h * dW - dZ)
        I_11  = 0.5 * (dW**2 - h)
        I_011 = 0.5 * h * I_11                    
        I_101 = np.sqrt(h**3/12.0) * np.random.randn(len(z0))   
        I_110 = 0.5 * h * I_11                     
        I_111 = (1/6) * (dW**3 - 3.0*h*dW)
        I_1111 = (1/24)*(dW**4 - 6.0*h*(dW**2) + 3.0*h**2)

        zs[i+1] = ( zs[i] + F(ts[i],zs[i])*h + G(ts[i],zs[i])*dW 
                   + L0(ts[i],zs[i],F,G,F)*I_00 + L1(ts[i],zs[i],F,G,F)*I_01
                   + L0(ts[i],zs[i],F,G,G)*I_10 + L1(ts[i],zs[i],F,G,G)*I_11
                   + L1_L1(ts[i],zs[i],F,G,G)*I_011 + L1_L0(ts[i],zs[i],F,G,G)*I_101
                   + L0_L1(ts[i],zs[i],F,G,G)*I_110 + L1_L1(ts[i],zs[i],F,G,G)*I_111
                   + L1_L1_L1(ts[i],zs[i],F,G,G)*I_1111 )
        
        if progress:
            progress(i+1, N, ts[i+1], zs[i+1])
        elif verbose and (((i+1) % print_every) == 0 or i+1 == N):
            print(f"[ito2] step {i+1}/{N}", end="\r", flush=True)
    if verbose:
        print()
    return ts,zs