import numpy as np
import matplotlib.pyplot as plt
from sdesim.integrators.rk4 import RK

def F(t,y):    # Define the vectorised ODE
    p = t % 364
    
    X, sig2 = 14.4, 1.5  # Food eclosion efficiency & Max. feedback rate
    d, sig3 = 0.016, 51  # Bee pollination factor & Bee pollination efficiency
    eps, sig4 = 105, 51  # Other pollination factor & Other pollination efficiency
    l, C = 1, 0.2        # Forager efficiency & Plant death rate
    a, w = 0.022, 0.33   # Food production rate & Food decay rate


    if 0 <= p < 92:
        b, K = 500, 8000    # Max. eclostion rate & Brood maintenance coefficient
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
    elif 92 <= p < 184:
        b, K = 1500, 12000    # Max. eclostion rate & Brood maintenance coefficient
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
    elif 184 <= p < 275:
        b, K = 500, 8000    # Max. eclostion rate & Brood maintenance coefficient
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
    elif 275 <= p < 364:
        b, K = 0, 6000    # Max. eclostion rate & Brood maintenance coefficient
        sig1, sig2 = 0, 1.5  # Base recruitment rate
        mu, phi    = 0.00649, 0  # Hive death rate & Forager death rate
    return np.array([b*(y[3]**2 / (X**2 + y[3]**2))*((y[0]+y[1])**2 / (K**2 + (y[0]+y[1])**2)) - sig1*y[0] + sig2*((y[1])**2 / (y[0]+y[1])) - mu*y[0],
                     sig1*y[0] - sig2*((y[1])**2 / (y[0]+y[1])) - phi*y[1],
                     d*y[1]*sig3*(y[3]**2 / (l + y[3]**2)) + eps*sig4*(y[3]**2 / (l + y[3]**2))-C*y[2],
                     a*y[2]-d*y[1]*(y[3]**2 / (l + y[3]**2)) - eps*(y[3]**2 / (l + y[3]**2)) - w*y[3]])

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

print(ys[:,2])

plt.plot(ts,ys[:,2],label = 'N')
#plt.plot(ts,ys[:,3],label = 'F')
plt.xlabel('t-axis(Days)')
plt.ylabel('Population')
plt.title('A plot of the numerical solution for Plant Population')
plt.legend()
plt.show()

plt.plot(x_numeric,y_numeric)
plt.xlabel('B_1-axis')
plt.ylabel('B_2-axis')
plt.title('A plot of the numerical phase portrait')
plt.show()

