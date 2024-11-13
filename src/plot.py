from src.util import fourier_eval, align_fourier, error
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def plot_fourier_curve(c, show_start=True, **kwargs):
    # Plot the curve
    t = jnp.linspace(0, 2*jnp.pi, 500)
    z = fourier_eval(t, c)        
    l = plt.plot(z.real, z.imag, **kwargs)
    if show_start:
        plt.scatter(z[0].real, z[0].imag, color=l[0].get_color())
        
def streamplot(X, Y, U, V, *args, **kwargs):
    X = np.array(X)
    Y = np.array(Y)
    U = np.array(U)
    V = np.array(V)
    plt.streamplot(X, Y, U, V, *args, **kwargs)
    
    
def plot_corr_table(width, clist, Nref=40):
    for i in range(width):
        for j in range(i+1):
            col1 = "red" if i % 2 == 0 else "blue"
            col2 = "red" if j % 2 == 0 else "blue"
            c1 = clist[i]
            c2 = clist[j]
            c21 = align_fourier(c2, c1, Nref=Nref)        
            dist = error(c1, c21)
            
            plt.subplot(width, width, i*width+j+1)
            plot_fourier_curve(c1, color=col1)
            plot_fourier_curve(c21, color=col2)
            plt.axis("off")
            
            ax = plt.subplot(width, width, j*width+i+1)
            # Color background according to distance
            ax.set_facecolor(plt.cm.viridis(dist/0.01))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])