import numpy as np

#Evaluate derivatives via order 4 finite differences

h1 = np.finfo(float).eps**(1/5)
h2 = np.finfo(float).eps**(1/6)
h3 = np.finfo(float).eps**(1/7)

def dtf(t,z,f):
        grad = (f(t-2*h1,z) - f(t+2*h1,z) + 8*(f(t+h1,z)-f(t-h1,z))) / (12*h1)
        return grad

def dttf(t,z,f):
        grad = (- f(t+2*h2,z)+ 16*f(t+h2,z) - 30*f(t,z) + 16*f(t-h2,z) - f(t-2*h2,z)) / (12*h2**2)
        return grad

def dxf(t,z,f):
        d = len(f(t,z))
        grad = np.zeros(d)
        I = np.eye(len(z))
        for i in range(len(z)):
            eps = I[i]
            grad[i] = (f(t,z-2*h1*eps)[i] - f(t,z+2*h1*eps)[i] + 8*(f(t,z+h1*eps)[i] - f(t,z-h1*eps)[i])) / (12*h1)
        return grad

def dxxf(t,z,f):
        d = len(f(t,z))
        grad = np.zeros(d)
        I = np.eye(len(z))
        for i in range(len(z)):
            eps = I[i]
            grad[i] = (- f(t,z+2*h2*eps)[i]+ 16*f(t,z+h2*eps)[i] - 30*f(t,z)[i] + 16*f(t,z-h2*eps)[i] - f(t,z-2*h2*eps)[i]) / (12*h2**2)
        return grad

def dxxxf(t,z,f):
        d = len(f(t,z))
        grad = np.zeros(d)
        I = np.eye(len(z))
        for i in range(len(z)):
            eps = I[i]
            grad[i] = (f(t,z-3*h3*eps)[i] + 8*f(t,z+2*h3*eps)[i] - 13*f(t,z+h3*eps)[i] + 13*f(t,z-h3*eps)[i] - 8*f(t,z-2*h3*eps)[i] - f(t,z+3*h3*eps)[i]) / (8*h3**3)
        return grad

def dtxf(t,z,f):
        d = len(f(t,z))
        grad = np.zeros(d)
        I = np.eye(len(z))

        w = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}

        ht = np.finfo(float).eps**(1/5) * max(1.0, abs(t))
        for i in range(len(z)):
            eps = I[i]
            hx = np.finfo(float).eps**(1/5) * max(1.0, abs(z[i]))
            grad[i] = 0
            for n in (-2,-1,1,2):
                for m in (-2,-1,1,2):
                    grad[i] += w[n]*w[m]*f(t+n*ht, z+hx*m*eps)[i]
            grad[i] = grad[i]/(144*ht*hx)
        return grad

def dtxxf(t, z, f):
    d = len(f(t, z))
    grad = np.zeros(d)
    I = np.eye(len(z))

    wt = {-2: 1, -1: -8, 1: 8, 2: -1}      
    wx = {-2: -1, -1: 16, 0: -30, 1: 16, 2: -1}

    ht = np.finfo(float).eps**(1/5) * max(1.0, abs(t))
    for i in range(len(z)):
        eps = I[i]
        hx = np.finfo(float).eps**(1/6) * max(1.0, abs(z[i]))
        grad[i] = 0
        for n in (-2,-1,1,2):
            for m in (-2, -1, 0, 1, 2):
                grad[i] += wt[n] * wx[m] * f(t + n*ht, z + m*hx*eps)[i]
        grad[i] = grad[i]/(144 * ht * hx*hx)
    return grad
