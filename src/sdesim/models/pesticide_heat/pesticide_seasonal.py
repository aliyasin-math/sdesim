import numpy as np
import matplotlib.pyplot as plt
from sdesim.integrators.rk4 import RK
from sdesim.models.stochastic_params import K, d, sig2, a, w

def F(t,y):    #Define the vectorised ODE
    p = t % 364
    
    X = 14.4  # Food eclosion efficiency
    sig3 = 51  # Bee pollination efficiency
    eps, sig4 = 105, 51  # Other pollination factor & Other pollination efficiency
    l, C = 1, 0.14        # Forager efficiency & Plant death rate
    LD50 = 10**(-6)  # Median lethal dose & Half-life of pesticide 
    T = 40

    if 0 <= p < 92:    
        b = 500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
        return np.array([b*(y[3]**2 / (X**2 + y[3]**2))*((y[0]+y[1])**2 / (K(t)**2 + (y[0]+y[1])**2)) - sig1*y[0] + sig2(t)*((y[1])**2 / (y[0]+y[1])) - mu*y[0],
                     sig1*y[0] - sig2(t)*((y[1])**2 / (y[0]+y[1])) - phi*y[1],
                     d(t)*y[1]*sig3*(y[3]**2 / (l + y[3]**2)) + eps*sig4*(y[3]**2 / (l + y[3]**2))-C*y[2],
                     a(t)*y[2]-d(t)*y[1]*(y[3]**2 / (l + y[3]**2)) - eps*(y[3]**2 / (l + y[3]**2)) - w(t)*y[3]])
    elif 92 <= p < 184:
        b = 1500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
        g, m = 0.20, 0.50  # Food consumed daily by a hive bee & Food collected daily by a forager
        c = 1.9*(10**(-7))   # Maximum concentration of pesticide
        t_1 = t % 42
        X1 = (2**(- t_1/T))*g*c 
        X2 = (2**(- t_1/T))*m*c
        return np.array([b*(y[3]**2 / (X**2 + y[3]**2))*((y[0]+y[1])**2 / (K(t)**2 + (y[0]+y[1])**2)) - sig1*y[0] + sig2(t)*((y[1])**2 / (y[0]+y[1])) - mu*y[0] - ((X1*y[0])/(X1 + LD50)),
                     sig1*y[0] - sig2(t)*((y[1])**2 / (y[0]+y[1])) - phi*y[1] - ((X2*y[1])/(X2 + LD50)),
                     d(t)*y[1]*sig3*(y[3]**2 / (l + y[3]**2)) + eps*sig4*(y[3]**2 / (l + y[3]**2))-C*y[2],
                     a(t)*y[2]-d(t)*y[1]*(y[3]**2 / (l + y[3]**2)) - eps*(y[3]**2 / (l + y[3]**2)) - w(t)*y[3]])
    elif 184 <= p < 275:
        b = 500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
        return np.array([b*(y[3]**2 / (X**2 + y[3]**2))*((y[0]+y[1])**2 / (K(t)**2 + (y[0]+y[1])**2)) - sig1*y[0] + sig2(t)*((y[1])**2 / (y[0]+y[1])) - mu*y[0],
                     sig1*y[0] - sig2(t)*((y[1])**2 / (y[0]+y[1])) - phi*y[1],
                     d(t)*y[1]*sig3*(y[3]**2 / (l + y[3]**2)) + eps*sig4*(y[3]**2 / (l + y[3]**2))-C*y[2],
                     a(t)*y[2]-d(t)*y[1]*(y[3]**2 / (l + y[3]**2)) - eps*(y[3]**2 / (l + y[3]**2)) - w(t)*y[3]])
    elif 275 <= p < 364:
        b = 0    # Max. eclostion rate
        sig1 = 0  # Base recruitment rate
        mu, phi    = 0.00649, 0  # Hive death rate & Forager death rate
        return np.array([b*(y[3]**2 / (X**2 + y[3]**2))*((y[0]+y[1])**2 / (K(t)**2 + (y[0]+y[1])**2)) - sig1*y[0] + sig2(t)*((y[1])**2 / (y[0]+y[1])) - mu*y[0],
                     sig1*y[0] - sig2(t)*((y[1])**2 / (y[0]+y[1])) - phi*y[1],
                     d(t)*y[1]*sig3*(y[3]**2 / (l + y[3]**2)) + eps*sig4*(y[3]**2 / (l + y[3]**2))-C*y[2],
                     a(t)*y[2]-d(t)*y[1]*(y[3]**2 / (l + y[3]**2)) - eps*(y[3]**2 / (l + y[3]**2)) - w(t)*y[3]])

y0 = np.array([15000, 0, 1000, 1000])

# Choose number of steps N and step size h
N, h = 5*3640, 0.1

ts, ys = RK(y0, N, h, F)
x_numeric = ys[:,0]
y_numeric = ys[:,1]

plt.plot(ts,ys[:,0],label = 'H')
plt.plot(ts,ys[:,1],label = 'F')
plt.xlabel('t-axis(Days)')
plt.ylabel('Population')
plt.title('A plot of the numerical solutions for Hive and Forager Bees')
plt.legend()
plt.show()

plt.plot(x_numeric,y_numeric)
plt.xlabel('B_1-axis')
plt.ylabel('B_2-axis')
plt.title('A plot of the numerical phase portrait')
plt.show()