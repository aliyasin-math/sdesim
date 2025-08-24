import numpy as np
import matplotlib.pyplot as plt
from sdesim.integrators.ito2 import ito2
from sdesim.integrators.rk4 import RK
from tqdm import tqdm

# Define the functions
def F1(t,y):
    p = t % 364

    d, sig2 = 0.016, 1.5  # Bee pollination factor & Max. feedback rate
    a, w = 0.022, 0.33   # Food production rate & Food decay rate

    if 0 <= p < 92:
        K = 8000    # Spring Brood maintenance coefficient
    elif 92 <= p < 184:
        K = 12000    # Summer Brood maintenance coefficient
    elif 184 <= p < 275:
        K = 8000    # Autumn Brood maintenance coefficient
    elif 275 <= p < 364:
        K = 6000    # Winter maintenance coefficient
    return np.array([th[0]*(K-y[0]),
                     th[1]*(d-y[1]),
                     th[2]*(sig2-y[2]),
                     th[3]*(a-y[3]),
                     th[4]*(w-y[4])])

def G1(t,z):
    return np.array([s[0],
                     s[1],
                     s[2],
                     s[3],
                     s[4]])

# Define the SDE parameters
th = np.array([1,1,1,1,1])
s = np.array([1600,0.0032,0.30,0.0044,0.066])

z0  = np.array([8000, 0.016, 1.5, 0.022, 0.33])

# Choose number of steps N and step size h so that t runs over [0,12]
N, h = 18190, 0.1

with tqdm(total=2*N, desc="ito2", dynamic_ncols=True) as pbar:
    def on_step(i, N_total, t, z):
        pbar.update(1)
        if i % 500 == 0 or i == N_total:
            pbar.set_postfix(step=i, t=f"{t:.1f}")
    ts, zs = ito2(z0, 2*N, h/2, F1, G1, progress=on_step)

plt.plot(ts,zs[:,0],label = 'K')
plt.xlabel('t-axis (Days)')
plt.ylabel('Brood Maintainence Coefficient')
plt.title('A plot of numerical solution paths for the Noisy Brood Maintainence Coefficient')
plt.legend()
plt.show()

plt.plot(ts,zs[:,1],label = 'd', color = 'coral')
plt.xlabel('t-axis (Days)')
plt.ylabel('Bee pollination factor')
plt.title('A plot of numerical solution paths for the Noisy Bee pollination factor')
plt.legend()
plt.show()

plt.plot(ts,zs[:,2],label = 'sig2', color = 'red')
plt.xlabel('t-axis (Days)')
plt.ylabel('Max. feedback rate')
plt.title('A plot of numerical solution paths for the Noisy Max. feedback rate')
plt.legend()
plt.show()

plt.plot(ts,zs[:,3],label = 'a', color = 'purple')
plt.xlabel('t-axis (Days)')
plt.ylabel('Food production rate')
plt.title('A plot of numerical solution paths for the Noisy Food production rate')
plt.legend()
plt.show()

plt.plot(ts,zs[:,4],label = 'w', color = 'cyan')
plt.xlabel('t-axis (Days)')
plt.ylabel('Food decay rate')
plt.title('A plot of numerical solution paths for the Noisy Food decay rate')
plt.legend()
plt.show()

def f(t,t0,L):
    if  t < t0+L/3:
        return np.exp(-(t-(t0+L/3))**2 / (2*(L/9)**2))
    elif  t0+L/3 <= t:
        return np.exp(-(t-(t0+L/3))**2 / (2*(2*L/9)**2))


def K(t):
    return np.interp(t, ts, zs[:,0])

def d(t):
    if 484 <= t < 527:
        return np.interp(t, ts, zs[:,1]) - 0.0032*f(t,484,42) 
    else: 
        return np.interp(t, ts, zs[:,1])

def sig2(t):
    if 484 <= t < 527:
        return np.interp(t, ts, zs[:,2]) - 0.3*f(t,484,42) 
    else:
        return np.interp(t, ts, zs[:,2])

def a(t):
    return np.interp(t, ts, zs[:,3])

def w(t):
    return np.interp(t, ts, zs[:,4])

def F(t,y):    #Define the vectorised ODE
    p = t % 364
    
    X = 14.4  # Food eclosion efficiency
    sig3 = 51  # Bee pollination efficiency
    eps, sig4 = 105, 51  # Other pollination factor & Other pollination efficiency
    C = 0.14        # Plant death rate

    if 0 <= p < 92:
        b = 500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
        l = 1       # Forager efficiency
    elif 92 <= p < 184 and (484 > t  or t >= 527):
        b = 1500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
        l = 1       # Forager efficiency
    elif 484 <= t < 527:
        b = 1500- 300*f(t,484,42)   # Max. eclostion rate
        sig1 = 0.25 - 0.05*f(t,484,42)  # Base recruitment rate
        mu = 0 + 0.15*f(t,484,42)         # Hive death rate (Disease)
        phi = 0.08511 + 0.15*f(t,484,42)  # Forager death rate (Disease)
        l = 1 + 0.2*f(t,484,42)
    elif 184 <= p < 275:
        b = 500    # Max. eclostion rate
        sig1 = 0.25  # Base recruitment rate
        mu, phi    = 0, 0.08511  # Hive death rate & Forager death rate
        l = 1       # Forager efficiency
    elif 275 <= p < 364: 
        b = 0    # Max. eclostion rate
        sig1 = 0  # Base recruitment rate
        mu, phi    = 0.00649, 0  # Hive death rate & Forager death rate
        l = 1       # Forager efficiency
    return np.array([b*(y[3]**2 / (X**2 + y[3]**2))*((y[0]+y[1])**2 / (K(t)**2 + (y[0]+y[1])**2)) - sig1*y[0] + sig2(t)*((y[1])**2 / (y[0]+y[1])) - mu*y[0],
                     sig1*y[0] - sig2(t)*((y[1])**2 / (y[0]+y[1])) - phi*y[1],
                     d(t)*y[1]*sig3*(y[3]**2 / (l + y[3]**2)) + eps*sig4*(y[3]**2 / (l + y[3]**2))-C*y[2],
                     a(t)*y[2]-d(t)*y[1]*(y[3]**2 / (l + y[3]**2)) - eps*(y[3]**2 / (l + y[3]**2)) - w(t)*y[3]])

y0 = np.array([10000, 0, 1000, 1000])

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