import numpy as np
import matplotlib.pyplot as plt
from sdesim.integrators.ito2 import ito2
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

# Choose number of steps N and step size h
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

def K(t):
    return np.interp(t, ts, zs[:,0])

def d(t):
    return np.interp(t, ts, zs[:,1])

def sig2(t):
    return np.interp(t, ts, zs[:,2])

def a(t):
    return np.interp(t, ts, zs[:,3])

def w(t):
    return np.interp(t, ts, zs[:,4])