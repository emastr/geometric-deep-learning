from jax import jit
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft
import numpy as np


## Signal processing
def rotate(z, angle, shift=0):
    return jnp.exp(1j*angle)*jnp.roll(z, shift)

def rotate_fourier(c, angle, shift):
    K = (len(c)-1)//2
    ks = jnp.arange(-K, K+1)
    return jnp.exp(1j * angle - 1j * ks[:,None]*shift) * c

def normalize(z):
    z = z - jnp.mean(z)
    return z / jnp.sum(jnp.abs(z)**2)**0.5

def normalize_fft(c):
    K = (len(c)-1)//2
    c = c.at[K].set(0)
    c = c / jnp.linalg.norm(c)
    return c

def flip(x):
    return jnp.flip(x, axis=0)  

def corr_fft(x1, x2):
    return flip(ifft(fft(x1.conj().T) * fft(flip(x2).T)).T)


### Fourier Series
def fourier_eval(t, c, deriv=0):
    K = (len(c)-1)//2
    ks = jnp.arange(-K, K+1)
    E = jnp.exp(1j * ks[None, :] * t[:, None])
    c = (1j*ks[:, None])**deriv * c
    return E @ c


def fit_fourier(t, z, K=10):
    if len(z.shape) == 1:
        z = z[:, None]
    ks = jnp.arange(-K, K+1)
    E = jnp.exp(1j * ks[None, :] * t[:, None])        
    c = jnp.linalg.solve(jnp.conjugate(E.T) @ E, jnp.conjugate(E.T) @ z)
    return c


### Data
def loop_erased_rw():
    N = 100
    Nmax = 200
    Kmax = 20
    x,y = [0],[0]
    steps = [(0,1),(0,-1),(-1,0),(1,0)]

    for n in range(100000):
        step_idx = np.random.randint(0,4)
        dx,dy = steps[step_idx]
        xn = x[-1]+dx
        yn = y[-1]+dy

        intersects = [i for i in range(len(y)-1) if x[i]==xn and y[i]==yn]
        assert len(intersects) <= 1, "Something is wrong"

        if len(intersects) == 0:
            x.append(xn)
            y.append(yn)

        elif len(intersects) == 1:
            i = intersects[0]
            if (len(x) - i) >= Nmax:
                x = x[i:]
                y = y[i:]
                break
            else:
                x = x[:i+1]
                y = y[:i+1]        
    
    # Define numpy arrays
    z = jnp.array(x) + 1j*np.array(y)
    dz = z - jnp.roll(z, 1)
    
    # If clock-wise, change direction.
    dz0 = dz[1]/jnp.abs(dz[1])
    z0 = (z[0]+z[1])/2
    dz0_orth = 1j*dz0
    dot = lambda x,y: jnp.real(x*y.conj())
    count = 0
    
    #plt.figure()
    for i in range(1,len(z)-1):
        a = z[i]-z0
        b = z[i+1] - z0
        if (dot(a, dz0) * dot(b, dz0) < 0) and (dot((a+b)/2, dz0_orth) >= 0): 
            count += 1
    if count % 2 == 0:
        z = z[::-1]
        
    # Compute arc_lengths
    arclen = jnp.abs(dz)
    totlen = jnp.sum(arclen)
    t = 2*jnp.pi * jnp.cumsum(arclen)/totlen

    # Fit
    c = fit_fourier(t, z, K=Kmax)
    c = normalize_fft(c)
    return c

###  Alignment 
def error(z1, z2):
    return (jnp.mean(jnp.abs(z1-z2)**2))**0.5


def distance(c0, c1, Nref=100):
    c21 = align_fourier(c0, c1, Nref)
    return error(c21, c1)

@jit
def align(z0, z1):
    ab = corr_fft(z0, z1)
    m = jnp.argmax(jnp.abs(ab))
    theta = jnp.arctan2(ab[m].imag, ab[m].real)
    return theta, m


def align_fourier(c0, c1, Nref=100):
    t = jnp.linspace(0, 2*jnp.pi, Nref+1)[1:]
    z0 = fourier_eval(t, c0)
    z1 = fourier_eval(t, c1)
    theta, m = align(z0, z1)
    return rotate_fourier(c0, theta, t[m-1])
    
    
    

