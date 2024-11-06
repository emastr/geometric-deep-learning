from src.util import fourier_eval
import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_fourier_curve(c, show_start=True, **kwargs):
    # Plot the curve
    t = jnp.linspace(0, 2*jnp.pi, 500)
    z = fourier_eval(t, c)        
    l = plt.plot(z.real, z.imag, **kwargs)
    if show_start:
        plt.scatter(z[0].real, z[0].imag, color=l[0].get_color())