import numpy as np
import matplotlib.pyplot as plt
from sdesim.integrators.rk4 import RK
from sdesim.models.stochastic_params import K, d, sig2, a, w

def f(t,t0,L):
    return np.exp(-(t-(t0+L/2))**2 / (2*(L/6)**2))

def F(t,y):    #Define the vectorised ODE
    p = t % 364
    
    X = 14.4  # Food eclosion efficiency
    sig3 = 51  # Bee pollination efficiency
    eps, sig4 = 105, 51  # Other pollination factor & Other pollination efficiency
    l, C = 1, 0.14        # Forager efficiency & Plant death rate

    if 0 <= p < 92:
        b = 500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
    elif 92 <= p < 184 and (484 > t  or t >= 527):
        b = 1500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
    elif 484 <= t < 527:
        b = 1500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu = 0 + 0.15*f(t,484,42)         # Hive death rate (Disease)
        phi = 0.08511 + 0.15*f(t,484,42)  # Forager death rate (Disease)
    elif 184 <= p < 275:
        b = 500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
    elif 275 <= p < 364: 
        b = 0    # Max. eclostion rate
        sig1 = 0  # Base recruitment rate
        mu, phi    = 0.00649, 0  # Hive death rate & Forager death rate
    return np.array([b*(y[3]**2 / (X**2 + y[3]**2))*((y[0]+y[1])**2 / (K(t)**2 + (y[0]+y[1])**2)) - sig1*y[0] + sig2(t)*((y[1])**2 / (y[0]+y[1])) - mu*y[0],
                     sig1*y[0] - sig2(t)*((y[1])**2 / (y[0]+y[1])) - phi*y[1],
                     d(t)*y[1]*sig3*(y[3]**2 / (l + y[3]**2)) + eps*sig4*(y[3]**2 / (l + y[3]**2))-C*y[2],
                     a(t)*y[2]-d(t)*y[1]*(y[3]**2 / (l + y[3]**2)) - eps*(y[3]**2 / (l + y[3]**2)) - w(t)*y[3]])

y0 = np.array([10000, 0, 1000, 1000])

# Choose number of steps N and step size h so that t runs over [0,12]
N, h = 18190, 0.1

ts, ys = RK(y0, N, h, F)
x_numeric = ys[:,0]
y_numeric = ys[:,1]

plt.plot(ts,ys[:,0],label = 'B_1')
plt.plot(ts,ys[:,1],label = 'B_2')
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